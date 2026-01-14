#pragma once

namespace flash_attn_v2 {

template <typename T>
constexpr T constexpr_min(T&& a, T&& b) {
    return (a < b) ?: a : b;
}

}  // namespace flash_attn_v2