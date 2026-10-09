#include "core/animation/motion.h"

namespace serenity::animation {

// No default in either switch: a kind without a position or an extent fails
// to compile (-Wswitch, -Werror). .at() checks the record's index, which the
// scene reader guarantees; a broken guarantee throws rather than reads past.

contracts::Float3 position(const Motions& motions, MotionRecord record, frame::Seconds t) {
    switch (record.kind) {
    case MotionKind::wander:
        return position(motions.wanders.at(record.index), t);
    }
    return {};
}

Extent extent(const Motions& motions, MotionRecord record) {
    switch (record.kind) {
    case MotionKind::wander:
        return extent(motions.wanders.at(record.index));
    }
    return {};
}

}  // namespace serenity::animation
