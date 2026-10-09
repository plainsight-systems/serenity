#include "core/shapes/shapes.h"

#include <stdexcept>
#include <string>

namespace serenity::shapes {

std::vector<Bounds> bounds(const Shapes& shapes) {
    std::vector<Bounds> result;
    result.reserve(shapes.records.size());
    for (const PrimitiveRecord& record : shapes.records) {
        // No default: a kind without bounds fails to compile (-Wswitch,
        // -Werror). .at() checks the record's index, which the scene reader
        // guarantees; a broken guarantee throws rather than reads past.
        switch (record.kind) {
        case ShapeKind::sphere:
            result.push_back(bounds(shapes.spheres.at(record.index)));
            break;
        case ShapeKind::box:
            result.push_back(bounds(shapes.boxes.at(record.index)));
            break;
        }
    }
    return result;
}

}  // namespace serenity::shapes
