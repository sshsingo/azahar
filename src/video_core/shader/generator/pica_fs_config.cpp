// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <cstring>
#include <mutex>
#include <unordered_set>

#include "common/logging/log.h"
#include "video_core/shader/generator/pica_fs_config.h"

namespace Pica::Shader {

namespace {

using TevStageConfig = Pica::TexturingRegs::TevStageConfig;
using Source = TevStageConfig::Source;
using ColorModifier = TevStageConfig::ColorModifier;
using AlphaModifier = TevStageConfig::AlphaModifier;
using Operation = TevStageConfig::Operation;

// Set to false to remove the per draw overhead of RecordFSConfigStats.
constexpr bool EnableFSConfigStats = true;

// Must match the pass-through detection of the fragment shader generators.
bool IsPassThroughTevStage(const TevStageConfig& stage) {
    return stage.color_op == Operation::Replace && stage.alpha_op == Operation::Replace &&
           stage.color_source1 == Source::Previous && stage.alpha_source1 == Source::Previous &&
           stage.color_modifier1 == ColorModifier::SourceColor &&
           stage.alpha_modifier1 == AlphaModifier::SourceAlpha && stage.GetColorMultiplier() == 1 &&
           stage.GetAlphaMultiplier() == 1;
}

TevStageConfigRaw PassThroughTevStage() {
    TevStageConfig stage{};
    stage.color_source1.Assign(Source::Previous);
    stage.alpha_source1.Assign(Source::Previous);
    return {stage.sources_raw, stage.modifiers_raw, stage.ops_raw, stage.scales_raw};
}

u32 NumOperands(Operation op) {
    switch (op) {
    case Operation::Replace:
        return 1;
    case Operation::Modulate:
    case Operation::Add:
    case Operation::AddSigned:
    case Operation::Subtract:
    case Operation::Dot3_RGB:
    case Operation::Dot3_RGBA:
        return 2;
    default:
        // Lerp, MultiplyThenAdd, AddThenMultiply and unknown operations keep every operand.
        return 3;
    }
}

bool ReadsAlpha(ColorModifier modifier) {
    return modifier == ColorModifier::SourceAlpha || modifier == ColorModifier::OneMinusSourceAlpha;
}

bool ReadsAlpha(AlphaModifier modifier) {
    return modifier == AlphaModifier::SourceAlpha || modifier == AlphaModifier::OneMinusSourceAlpha;
}

void SetColorOperand(TevStageConfig& stage, u32 operand, Source source, ColorModifier modifier) {
    switch (operand) {
    case 0:
        stage.color_source1.Assign(source);
        stage.color_modifier1.Assign(modifier);
        break;
    case 1:
        stage.color_source2.Assign(source);
        stage.color_modifier2.Assign(modifier);
        break;
    default:
        stage.color_source3.Assign(source);
        stage.color_modifier3.Assign(modifier);
        break;
    }
}

void SetAlphaOperand(TevStageConfig& stage, u32 operand, Source source, AlphaModifier modifier) {
    switch (operand) {
    case 0:
        stage.alpha_source1.Assign(source);
        stage.alpha_modifier1.Assign(modifier);
        break;
    case 1:
        stage.alpha_source2.Assign(source);
        stage.alpha_modifier2.Assign(modifier);
        break;
    default:
        stage.alpha_source3.Assign(source);
        stage.alpha_modifier3.Assign(modifier);
        break;
    }
}

/// Liveness of the combiner registers used by the generated TEV code.
struct TevLiveness {
    bool output_rgb = true; ///< The final combiner output is always consumed
    bool output_a = true;
    bool buffer_rgb = false;
    bool buffer_a = false;
    bool next_buffer_rgb = false;
    bool next_buffer_a = false;
    u32 used_sources = 0;

    void Read(Source source, bool alpha) {
        switch (source) {
        case Source::Previous:
            (alpha ? output_a : output_rgb) = true;
            break;
        case Source::PreviousBuffer:
            (alpha ? buffer_a : buffer_rgb) = true;
            break;
        default:
            used_sources |= 1u << static_cast<u32>(source);
            break;
        }
    }
};

/**
 * Copies the used operands of one half (color or alpha) of a TEV stage into canon.
 * Stage 0 reads source3 in place of Previous, this is resolved here so the unused
 * operands can be dropped. If source3 itself is Previous it must be kept for that rule.
 */
template <typename Modifier, typename SetOperand>
void CanonicalizeOperands(u32 index, u32 num_operands, std::array<Source, 3> sources,
                          std::array<Modifier, 3> modifiers, TevStageConfig& canon,
                          TevLiveness& liveness, SetOperand&& set_operand) {
    bool reads_previous = false;
    for (u32 i = 0; i < num_operands; i++) {
        Source source = sources[i];
        if (index == 0 && source == Source::Previous) {
            source = sources[2];
        }
        reads_previous |= source == Source::Previous;
        set_operand(canon, i, source, modifiers[i]);
        liveness.Read(source, ReadsAlpha(modifiers[i]));
    }
    if (index == 0 && reads_previous && num_operands < 3) {
        set_operand(canon, 2, Source::Previous, Modifier{});
    }
}

} // Anonymous namespace

FramebufferConfig::FramebufferConfig(const Pica::RegsInternal& regs) {
    const auto& output_merger = regs.framebuffer.output_merger;
    scissor_test_mode.Assign(regs.rasterizer.scissor_test.mode);
    depthmap_enable.Assign(regs.rasterizer.depthmap_enable);
    shadow_rendering.Assign(regs.framebuffer.IsShadowRendering());
    alpha_test_func.Assign(output_merger.alpha_test.enable
                               ? output_merger.alpha_test.func.Value()
                               : Pica::FramebufferRegs::CompareFunc::Always);

    alphablend_enable.Assign(output_merger.alphablend_enable);
    requested_logic_op = output_merger.logic_op;

    logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);

    if (alphablend_enable) {
        requested_rgb_blend.eq = output_merger.alpha_blending.blend_equation_rgb.Value();
        requested_rgb_blend.src_factor = output_merger.alpha_blending.factor_source_rgb;
        requested_rgb_blend.dst_factor = output_merger.alpha_blending.factor_dest_rgb;

        requested_alpha_blend.eq = output_merger.alpha_blending.blend_equation_a.Value();
        requested_alpha_blend.src_factor = output_merger.alpha_blending.factor_source_a;
        requested_alpha_blend.dst_factor = output_merger.alpha_blending.factor_dest_a;
    }
}

void FramebufferConfig::ApplyProfile(const Profile& profile) {
    // Emulate logic op in the shader if needed and not supported.
    if (!profile.has_logic_op && !alphablend_enable) {
        logic_op.Assign(requested_logic_op);
    }

    if (!alphablend_enable) {
        return;
    }

    // Check if we don't need blend min/max emulation.
    if (profile.has_blend_minmax_factor) {
        requested_rgb_blend.SetMinMaxEmulationDisabled();
        requested_alpha_blend.SetMinMaxEmulationDisabled();
    } else if (profile.is_vulkan) {
        // VK_BLEND_OP_MIN/MAX ignore the blend factors, unlike the PICA hardware.
        if (requested_rgb_blend.UsesNoOpFactors()) {
            requested_rgb_blend.SetMinMaxEmulationDisabled();
        }
        if (requested_alpha_blend.UsesNoOpFactors()) {
            requested_alpha_blend.SetMinMaxEmulationDisabled();
        }
    }
}

TextureConfig::TextureConfig(const Pica::TexturingRegs& regs) {
    texture0_type.Assign(regs.texture0.type);
    texture2_use_coord1.Assign(regs.main_config.texture2_use_coord1 != 0);
    combiner_buffer_input.Assign(regs.tev_combiner_buffer_input.update_mask_rgb.Value() |
                                 regs.tev_combiner_buffer_input.update_mask_a.Value() << 4);
    fog_mode.Assign(regs.fog_mode);
    fog_flip.Assign(regs.fog_flip != 0);
    shadow_texture_orthographic.Assign(regs.shadow.orthographic != 0);

    const auto pica_textures = regs.GetTextures();
    for (u32 tex_index = 0; tex_index < 3; tex_index++) {
        requested_wrap[tex_index].s = pica_textures[tex_index].config.wrap_s;
        requested_wrap[tex_index].t = pica_textures[tex_index].config.wrap_t;

        texture_border_color[tex_index].enable_s.Assign(false);
        texture_border_color[tex_index].enable_t.Assign(false);
    }

    const auto& stages = regs.GetTevStages();
    for (std::size_t i = 0; i < tev_stages.size(); i++) {
        const auto& tev_stage = stages[i];
        tev_stages[i].sources_raw = tev_stage.sources_raw;
        tev_stages[i].modifiers_raw = tev_stage.modifiers_raw;
        tev_stages[i].ops_raw = tev_stage.ops_raw;
        tev_stages[i].scales_raw = tev_stage.scales_raw;
        if (tev_stage.color_op == Pica::TexturingRegs::TevStageConfig::Operation::Dot3_RGBA) {
            tev_stages[i].sources_raw &= 0xFFF;
            tev_stages[i].modifiers_raw &= 0xFFF;
            tev_stages[i].ops_raw &= 0xF;
        }
    }
}

void TextureConfig::ApplyProfile(const Profile& profile) {
    // Emulate custom border color if needed and not supported.
    if (profile.has_custom_border_color) {
        return;
    }

    for (u32 i = 0; i < 3; i++) {
        texture_border_color[i].enable_s.Assign(
            requested_wrap[i].s == Pica::TexturingRegs::TextureConfig::WrapMode::ClampToBorder);

        texture_border_color[i].enable_t.Assign(
            requested_wrap[i].t == Pica::TexturingRegs::TextureConfig::WrapMode::ClampToBorder);
    }
}

u32 TextureConfig::CanonicalizeTevStages() {
    TevLiveness liveness{};
    u32 buffer_input = combiner_buffer_input;

    // Walk the stages backwards, tracking which parts of the combiner output and buffers are
    // consumed later on. Each stage executes (in this order):
    //   combiner_output = f(sources)          (skipped for pass-through stages)
    //   combiner_buffer = next_combiner_buffer
    //   next_combiner_buffer.rgb = combiner_output.rgb (if enabled for the stage)
    //   next_combiner_buffer.a = combiner_output.a     (if enabled for the stage)
    for (s32 index = static_cast<s32>(tev_stages.size()) - 1; index >= 0; index--) {
        if (index < 4) {
            const u32 alpha_bit = 1u << (index + 4);
            if ((buffer_input & alpha_bit) && liveness.next_buffer_a) {
                liveness.next_buffer_a = false;
                liveness.output_a = true;
            } else {
                buffer_input &= ~alpha_bit;
            }

            const u32 rgb_bit = 1u << index;
            if ((buffer_input & rgb_bit) && liveness.next_buffer_rgb) {
                liveness.next_buffer_rgb = false;
                liveness.output_rgb = true;
            } else {
                buffer_input &= ~rgb_bit;
            }
        }

        liveness.next_buffer_rgb |= liveness.buffer_rgb;
        liveness.next_buffer_a |= liveness.buffer_a;
        liveness.buffer_rgb = false;
        liveness.buffer_a = false;

        const TevStageConfig stage = tev_stages[index];
        const bool dot3_rgba = stage.color_op == Operation::Dot3_RGBA;
        const bool need_color = liveness.output_rgb || (dot3_rgba && liveness.output_a);
        const bool need_alpha = liveness.output_a && !dot3_rgba;
        if (IsPassThroughTevStage(stage) || (!need_color && !need_alpha)) {
            // The stage leaves combiner_output untouched, or nothing reads what it writes.
            tev_stages[index] = PassThroughTevStage();
            continue;
        }

        // The stage overwrites the whole combiner output. Unneeded halves are left zeroed,
        // which reads the primary color and can never be mistaken for a pass-through stage.
        TevStageConfig canon{};
        // The scales apply to the output channels, even when Dot3_RGBA fills the alpha channel.
        if (liveness.output_rgb && stage.GetColorMultiplier() != 1) {
            canon.color_scale.Assign(stage.color_scale.Value());
        }
        if (liveness.output_a && stage.GetAlphaMultiplier() != 1) {
            canon.alpha_scale.Assign(stage.alpha_scale.Value());
        }
        liveness.output_rgb = false;
        liveness.output_a = false;

        if (need_color) {
            canon.color_op.Assign(stage.color_op);
            CanonicalizeOperands<ColorModifier>(
                index, NumOperands(stage.color_op),
                {stage.color_source1, stage.color_source2, stage.color_source3},
                {stage.color_modifier1, stage.color_modifier2, stage.color_modifier3}, canon,
                liveness, SetColorOperand);
        }
        if (need_alpha) {
            canon.alpha_op.Assign(stage.alpha_op);
            CanonicalizeOperands<AlphaModifier>(
                index, NumOperands(stage.alpha_op),
                {stage.alpha_source1, stage.alpha_source2, stage.alpha_source3},
                {stage.alpha_modifier1, stage.alpha_modifier2, stage.alpha_modifier3}, canon,
                liveness, SetAlphaOperand);
        }
        tev_stages[index] = {canon.sources_raw, canon.modifiers_raw, canon.ops_raw,
                             canon.scales_raw};
    }

    combiner_buffer_input.Assign(buffer_input);
    return liveness.used_sources;
}

void TextureConfig::CanonicalizeTextureUnits(const std::array<bool, 4>& units_used) {
    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;
    using WrapMode = Pica::TexturingRegs::TextureConfig::WrapMode;

    if (!units_used[0]) {
        texture0_type.Assign(TextureType::Texture2D);
    }
    if (texture0_type != TextureType::Shadow2D && texture0_type != TextureType::ShadowCube) {
        shadow_texture_orthographic.Assign(0);
    }
    if (!units_used[2]) {
        texture2_use_coord1.Assign(0);
    }
    if (fog_mode != Pica::TexturingRegs::FogMode::Fog) {
        fog_flip.Assign(0);
    }

    // Only ClampToBorder is relevant to the shader, for border color emulation.
    for (u32 i = 0; i < requested_wrap.size(); i++) {
        auto& wrap = requested_wrap[i];
        if (!units_used[i] || wrap.s != WrapMode::ClampToBorder) {
            wrap.s = WrapMode::ClampToEdge;
        }
        if (!units_used[i] || wrap.t != WrapMode::ClampToBorder) {
            wrap.t = WrapMode::ClampToEdge;
        }
    }
}

LightConfig::LightConfig(const Pica::LightingRegs& regs) {
    if (regs.disable) {
        return;
    }

    enable.Assign(1);
    src_num.Assign(regs.max_light_index + 1);
    config.Assign(regs.config0.config);
    enable_primary_alpha.Assign(regs.config0.enable_primary_alpha);
    enable_secondary_alpha.Assign(regs.config0.enable_secondary_alpha);
    bump_mode.Assign(regs.config0.bump_mode);
    bump_selector.Assign(regs.config0.bump_selector);
    bump_renorm.Assign(regs.config0.disable_bump_renorm == 0);
    clamp_highlights.Assign(regs.config0.clamp_highlights != 0);

    enable_shadow.Assign(regs.config0.enable_shadow != 0);
    if (enable_shadow) {
        shadow_primary.Assign(regs.config0.shadow_primary != 0);
        shadow_secondary.Assign(regs.config0.shadow_secondary != 0);
        shadow_invert.Assign(regs.config0.shadow_invert != 0);
        shadow_alpha.Assign(regs.config0.shadow_alpha != 0);
        shadow_selector.Assign(regs.config0.shadow_selector);
    }

    for (u32 light_index = 0; light_index <= regs.max_light_index; ++light_index) {
        const u32 num = regs.light_enable.GetNum(light_index);
        const auto& light = regs.light[num];
        lights[light_index].num.Assign(num);
        lights[light_index].directional.Assign(light.config.directional != 0);
        lights[light_index].two_sided_diffuse.Assign(light.config.two_sided_diffuse != 0);
        lights[light_index].geometric_factor_0.Assign(light.config.geometric_factor_0 != 0);
        lights[light_index].geometric_factor_1.Assign(light.config.geometric_factor_1 != 0);
        lights[light_index].dist_atten_enable.Assign(!regs.IsDistAttenDisabled(num));
        lights[light_index].spot_atten_enable.Assign(!regs.IsSpotAttenDisabled(num));
        lights[light_index].shadow_enable.Assign(!regs.IsShadowDisabled(num));
    }

    lut_d0.enable.Assign(regs.config1.disable_lut_d0 == 0);
    if (lut_d0.enable) {
        lut_d0.abs_input.Assign(regs.abs_lut_input.disable_d0 == 0);
        lut_d0.type.Assign(regs.lut_input.d0.Value());
        lut_d0.SetScale(regs.lut_scale.GetScale(regs.lut_scale.d0));
    }

    lut_d1.enable.Assign(regs.config1.disable_lut_d1 == 0);
    if (lut_d1.enable) {
        lut_d1.abs_input.Assign(regs.abs_lut_input.disable_d1 == 0);
        lut_d1.type.Assign(regs.lut_input.d1.Value());
        lut_d1.SetScale(regs.lut_scale.GetScale(regs.lut_scale.d1));
    }

    // This is a dummy field due to lack of the corresponding register
    lut_sp.enable.Assign(1);
    lut_sp.abs_input.Assign(regs.abs_lut_input.disable_sp == 0);
    lut_sp.type.Assign(regs.lut_input.sp.Value());
    lut_sp.SetScale(regs.lut_scale.GetScale(regs.lut_scale.sp));

    lut_fr.enable.Assign(regs.config1.disable_lut_fr == 0);
    if (lut_fr.enable) {
        lut_fr.abs_input.Assign(regs.abs_lut_input.disable_fr == 0);
        lut_fr.type.Assign(regs.lut_input.fr.Value());
        lut_fr.SetScale(regs.lut_scale.GetScale(regs.lut_scale.fr));
    }

    lut_rr.enable.Assign(regs.config1.disable_lut_rr == 0);
    if (lut_rr.enable) {
        lut_rr.abs_input.Assign(regs.abs_lut_input.disable_rr == 0);
        lut_rr.type.Assign(regs.lut_input.rr.Value());
        lut_rr.SetScale(regs.lut_scale.GetScale(regs.lut_scale.rr));
    }

    lut_rg.enable.Assign(regs.config1.disable_lut_rg == 0);
    if (lut_rg.enable) {
        lut_rg.abs_input.Assign(regs.abs_lut_input.disable_rg == 0);
        lut_rg.type.Assign(regs.lut_input.rg.Value());
        lut_rg.SetScale(regs.lut_scale.GetScale(regs.lut_scale.rg));
    }

    lut_rb.enable.Assign(regs.config1.disable_lut_rb == 0);
    if (lut_rb.enable) {
        lut_rb.abs_input.Assign(regs.abs_lut_input.disable_rb == 0);
        lut_rb.type.Assign(regs.lut_input.rb.Value());
        lut_rb.SetScale(regs.lut_scale.GetScale(regs.lut_scale.rb));
    }
}

ProcTexConfig::ProcTexConfig(const Pica::TexturingRegs& regs) {
    if (!regs.main_config.texture3_enable) {
        return;
    }

    enable.Assign(1);
    coord.Assign(regs.main_config.texture3_coordinates);
    u_clamp.Assign(regs.proctex.u_clamp);
    v_clamp.Assign(regs.proctex.v_clamp);
    color_combiner.Assign(regs.proctex.color_combiner);
    alpha_combiner.Assign(regs.proctex.alpha_combiner);
    separate_alpha.Assign(regs.proctex.separate_alpha);
    noise_enable.Assign(regs.proctex.noise_enable);
    u_shift.Assign(regs.proctex.u_shift);
    v_shift.Assign(regs.proctex.v_shift);
    lut_width = regs.proctex_lut.width;
    lut_offset0 = regs.proctex_lut_offset.level0;
    lut_offset1 = regs.proctex_lut_offset.level1;
    lut_offset2 = regs.proctex_lut_offset.level2;
    lut_offset3 = regs.proctex_lut_offset.level3;
    lod_min = regs.proctex_lut.lod_min;
    lod_max = regs.proctex_lut.lod_max;
    lut_filter.Assign(regs.proctex_lut.filter);
}

FSConfig::FSConfig(const Pica::RegsInternal& regs, bool canonicalize)
    : framebuffer{regs}, texture{regs.texturing}, lighting{regs.lighting}, proctex{regs.texturing} {
    if (canonicalize) {
        Canonicalize();
    }
}

void FSConfig::Canonicalize() {
    if (framebuffer.alpha_test_func == Pica::FramebufferRegs::CompareFunc::Never) {
        // The shader discards every fragment, nothing else contributes to its output.
        std::memset(static_cast<void*>(&texture), 0, sizeof(texture));
        std::memset(static_cast<void*>(&lighting), 0, sizeof(lighting));
        std::memset(static_cast<void*>(&proctex), 0, sizeof(proctex));
        return;
    }

    const u32 used_sources = texture.CanonicalizeTevStages();
    const auto uses = [used_sources](Source source) {
        return ((used_sources >> static_cast<u32>(source)) & 1) != 0;
    };

    std::array<bool, 4> units_used{uses(Source::Texture0), uses(Source::Texture1),
                                   uses(Source::Texture2), uses(Source::Texture3)};

    // Lighting only produces the primary and secondary fragment colors.
    if (!uses(Source::PrimaryFragmentColor) && !uses(Source::SecondaryFragmentColor)) {
        std::memset(static_cast<void*>(&lighting), 0, sizeof(lighting));
    } else if (lighting.enable) {
        if (lighting.bump_mode != Pica::LightingRegs::LightingBumpMode::None) {
            units_used[lighting.bump_selector.Value()] = true;
        }
        if (lighting.enable_shadow) {
            units_used[lighting.shadow_selector.Value()] = true;
        }
    }

    // Procedural texture is only reachable through texture unit 3.
    if (!units_used[3]) {
        std::memset(static_cast<void*>(&proctex), 0, sizeof(proctex));
    }

    texture.CanonicalizeTextureUnits(units_used);
}

void RecordFSConfigStats(const Pica::RegsInternal& regs, const FSConfig& config) {
    if constexpr (!EnableFSConfigStats) {
        return;
    }

    static std::mutex mutex;
    static std::unordered_set<std::size_t> raw_hashes;
    static std::unordered_set<std::size_t> canonical_hashes;

    const std::size_t raw_hash = FSConfig{regs, false}.Hash();
    std::scoped_lock lock{mutex};
    const bool new_raw = raw_hashes.insert(raw_hash).second;
    const bool new_canonical = canonical_hashes.insert(config.Hash()).second;
    if (new_raw || new_canonical) {
        LOG_INFO(Render, "FSConfig stats: {} raw configs -> {} canonical configs",
                 raw_hashes.size(), canonical_hashes.size());
    }
}

} // namespace Pica::Shader
