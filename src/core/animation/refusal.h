#pragma once

#include <stdexcept>

namespace serenity::animation {

// Axis: Animation.
//
// What a motion kind throws when it cannot be made clear of the still shapes
// (contract 11, motion.h), and what the scene reader reports against the
// motion's line: a purpose-made type (E.14), so the reader never takes
// another std::invalid_argument, one an Obstacles implementation threw or a
// broken precondition's, for a refusal of the scene's numbers. A refusal is
// an argument the kind cannot make a motion of, so it is one of
// std::invalid_argument; numbers out of a kind's range, which the reader
// checks before it asks, are a plain std::invalid_argument.
class Refusal : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

}  // namespace serenity::animation
