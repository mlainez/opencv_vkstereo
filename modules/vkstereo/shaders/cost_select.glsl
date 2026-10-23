// Copyright 2016 Fixstars Corporation
// SPDX-License-Identifier: Apache-2.0
// Selects from the register-held vertical path plus both stored horizontal
// paths, as in select_path.glsl, with the same winner, uniqueness, subpixel
// and winner-key outputs.
struct SelectGeometry {
  int x;
  uint lane_disparity;
  uint base;
  // Candidates right of the image (d > x) never win: they carry a padding
  // term that exceeds every genuine sum, which also excludes them from the
  // second-best search. Winner keys and minima are read from unpadded lanes.
  uint pad[BLOCK];
};
SelectGeometry select_geometry(int x, uint lane, uint base) {
  SelectGeometry s;
  s.x = x;
  s.lane_disparity = lane * BLOCK;
  s.base = base;
  for (int k = 0; k < BLOCK; ++k) s.pad[k] = int(lane) * BLOCK + k <= x ? 0u : 0x8000u;
  return s;
}
// Both horizontal bytes stay below 128, so their packed sum has no carries.
PATH_WORD_TYPE add_words(PATH_WORD_TYPE a, PATH_WORD_TYPE b) { return a + b; }
uint pick(uint values[BLOCK], uint index) {
  uint t[BLOCK];
  for (int i = 0; i < BLOCK; ++i) t[i] = values[i];
#if BLOCK >= 16
  bool high8 = (index & 8u) != 0u;
  for (int i = 0; i < 8; ++i) t[i] = high8 ? t[i + 8] : t[i];
#endif
#if BLOCK >= 8
  bool high4 = (index & 4u) != 0u;
  for (int i = 0; i < 4; ++i) t[i] = high4 ? t[i + 4] : t[i];
#endif
  bool high2 = (index & 2u) != 0u;
  for (int i = 0; i < 2; ++i) t[i] = high2 ? t[i + 2] : t[i];
  return (index & 1u) != 0u ? t[1] : t[0];
}
void select_sum(uint vertical[BLOCK], int pixel, uint horizontal[BLOCK], SelectGeometry s);
void select_pixel(uint vertical[BLOCK], int pixel, int word, SelectGeometry s) {
  uint horizontal[BLOCK];
  for (int k = 0; k < BLOCK; k += PATH_PACK)
    unpack_lane(add_words(paths_forward[word + k / PATH_PACK], paths_reverse[word + k / PATH_PACK]), horizontal, k);
  select_sum(vertical, pixel, horizontal, s);
}
void select_sum(uint vertical[BLOCK], int pixel, uint horizontal[BLOCK], SelectGeometry s) {
  uint values[BLOCK];
  for (int k = 0; k < BLOCK; ++k) values[k] = vertical[k] + horizontal[k] + s.pad[k];
  uint best = 0xffffffffu;
  for (int k = 0; k < BLOCK; ++k) best = min(best, (values[k] << 16) | (s.lane_disparity + uint(k)));
  best = group_min(best);
  int winner = int(best & 65535u);
  uint minimum = best >> 16;
  // Entries within one disparity of the winner are excluded from second place.
  uint second = 0xffffffffu;
  uint near = s.lane_disparity + 1u - uint(winner);
  for (int k = 0; k < BLOCK; ++k) second = min(second, near + uint(k) <= 2u ? 0xffffffffu : values[k]);
  second = group_min(second);
  int lower_d = max(winner - 1, 0), upper_d = min(winner + 1, DISPARITIES - 1);
  uint lower = pick(values, uint(lower_d) % BLOCK);
  uint upper = pick(values, uint(upper_d) % BLOCK);
  // Every active lane executes the shuffles before the single-lane store.
  lower = subgroupShuffle(lower, s.base + uint(lower_d) / BLOCK);
  upper = subgroupShuffle(upper, s.base + uint(upper_d) / BLOCK);
  bool accepted = float(second) > float(minimum) * UNIQUENESS;
  float lo = float(lower), hi = float(upper);
  float denominator = lo + hi - 2.0 * float(minimum);
  float correction = clamp((lo - hi) / (2.0 * max(denominator, 1.0)), -0.5, 0.5);
  correction = mix(0.0, correction, denominator > 0.0);
  correction = mix(0.0, correction, winner > 0);
  correction = mix(0.0, correction, winner < min(DISPARITIES - 1, s.x));
  float result = float(winner) + correction;
  if (s.lane_disparity == 0u) {
    left_disparity[pixel] = mix(-1.0, result, accepted);
    winners[pixel] = mix(0xffffffffu, best, accepted);
  }
}
