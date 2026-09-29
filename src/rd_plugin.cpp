// SPDX-License-Identifier: GPL-3.0-or-later
// rd_plugin.cpp -- the dither registry.  See include/rd_plugin.h for the rationale.

#include "rd_plugin.h"

#include <cstring>
#include <mutex>

namespace rd {
namespace {

struct Registry {
  std::mutex mtx;
  std::vector<DitherAlgorithm> algos;
};

// Function-local static rather than a namespace-scope one: a plugin's static
// initialiser can run before this translation unit's globals are constructed, and
// the order is unspecified across translation units.  A local static is guaranteed
// to be constructed on first use, which makes registration order irrelevant.
Registry& Reg() {
  static Registry r;
  return r;
}

}  // namespace

bool RegisterDither(const DitherAlgorithm& algo) {
  if (algo.name == nullptr || algo.name[0] == '\0' || algo.run == nullptr) return false;
  Registry& r = Reg();
  std::lock_guard<std::mutex> lock(r.mtx);
  // Refuse duplicates rather than overwriting.  A plugin that shadows a built-in
  // silently is the kind of thing that costs an afternoon: --dither riemersma
  // quietly running someone else's code.
  for (const DitherAlgorithm& a : r.algos) {
    if (std::strcmp(a.name, algo.name) == 0) return false;
  }
  r.algos.push_back(algo);
  return true;
}

const DitherAlgorithm* FindDither(const char* name) {
  if (name == nullptr) return nullptr;
  Registry& r = Reg();
  std::lock_guard<std::mutex> lock(r.mtx);
  for (const DitherAlgorithm& a : r.algos) {
    if (std::strcmp(a.name, name) == 0) return &a;
  }
  return nullptr;
}

const std::vector<DitherAlgorithm>& RegisteredDithers() { return Reg().algos; }

std::string DitherListText() {
  Registry& r = Reg();
  std::lock_guard<std::mutex> lock(r.mtx);
  std::string out;
  for (std::size_t i = 0; i < r.algos.size(); ++i) {
    const DitherAlgorithm& a = r.algos[i];
    if (i) out += "\n";
    // Padded to a fixed width so the descriptions line up in a terminal.  16 is the
    // longest name below plus slack; a longer name simply pushes its own row out
    // rather than breaking the rest.
    const std::string pad(16 > std::strlen(a.name) ? 16 - std::strlen(a.name) : 1, ' ');
    out += std::string("  ") + a.name + pad + (a.description ? a.description : "");
  }
  return out;
}

}  // namespace rd
