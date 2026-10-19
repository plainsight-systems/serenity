#include "measure/commands.h"

#include <filesystem>
#include <format>
#include <string>
#include <string_view>

#include "core/contracts/linear_image.h"
#include "core/measurement/error.h"
#include "core/measurement/reference.h"
#include "core/output/pfm.h"

namespace serenity::measure {

std::string csv_field(std::string_view field) {
  if (field.find_first_of(",\"\r\n") == std::string_view::npos) {
    return std::string{field};
  }
  std::string quoted = "\"";
  for (const char c : field) {
    quoted += c;
    if (c == '"') {
      quoted += '"';
    }
  }
  return quoted + "\"";
}

std::string run(const MakeReference& command) {
  measurement::ReferenceBuilder builder;
  for (const std::filesystem::path& batch : command.batches) {
    // One batch held at a time: read, folded in, dropped (reference.h).
    builder.add(output::read_pfm(batch));
  }
  // The floors before the file, so nothing after the write can fail but
  // printing.
  const double mse_floor = builder.mse_floor();
  const double relative_mse_floor = builder.relative_mse_floor();
  // Created, never written over: the exclusive create is the check that
  // --out names nothing, so two runs racing to one file cannot both write
  // it (measure/options.h, core/output/pfm.h).
  output::write_pfm(command.out, builder.mean());
  return std::format(
      "batches {}\nmse_floor {:.9g}\nrelative_mse_floor {:.9g}\n",
      builder.batches(), mse_floor, relative_mse_floor);
}

std::string run(const MeasureError& command) {
  const contracts::LinearImage reference = output::read_pfm(command.reference);
  std::string table = "image,mse,relative_mse\n";
  for (const std::filesystem::path& image : command.images) {
    const measurement::ImageError error = measurement::error_against(
        {.image = output::read_pfm(image), .reference = reference});
    table += std::format("{},{:.9g},{:.9g}\n", csv_field(image.string()),
                         error.mse, error.relative_mse);
  }
  return table;
}

}  // namespace serenity::measure
