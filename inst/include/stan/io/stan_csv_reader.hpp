#ifndef STAN_IO_STAN_CSV_READER_HPP
#define STAN_IO_STAN_CSV_READER_HPP

#include <boost/algorithm/string.hpp>
#include <stan/math/prim.hpp>
#include <cctype>
#include <istream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
namespace io {

inline void prettify_stan_csv_name(std::string& variable) {
  if (variable.find_first_of(":.") != std::string::npos) {
    std::vector<std::string> parts;
    boost::split(parts, variable, boost::is_any_of(":"));
    for (auto& part : parts) {
      int pos = part.find('.');
      if (pos > 0) {
        part[pos] = '[';
        std::replace(part.begin(), part.end(), '.', ',');
        part += "]";
      }
    }
    variable = boost::algorithm::join(parts, ".");
  }
}

struct stan_csv_metadata {
  int stan_version_major;
  int stan_version_minor;
  int stan_version_patch;

  std::string model;
  std::string data;
  std::string init;
  size_t chain_id;
  size_t seed;
  bool random_seed;
  size_t num_samples;
  size_t num_warmup;
  bool save_warmup;
  size_t thin;
  bool append_samples;
  std::string method;
  std::string algorithm;
  std::string engine;
  int max_depth;

  stan_csv_metadata()
      : stan_version_major(0),
        stan_version_minor(0),
        stan_version_patch(0),
        model(""),
        data(""),
        init(""),
        chain_id(1),
        seed(0),
        random_seed(false),
        num_samples(0),
        num_warmup(0),
        save_warmup(false),
        thin(1),
        append_samples(false),
        method(""),
        algorithm(""),
        engine(""),
        max_depth(10) {}
};

struct stan_csv_adaptation {
  double step_size;
  Eigen::MatrixXd metric;

  stan_csv_adaptation() : step_size(0), metric(0, 0) {}
};

struct stan_csv_timing {
  double warmup;
  double sampling;

  stan_csv_timing() : warmup(0), sampling(0) {}
};

struct stan_csv {
  stan_csv_metadata metadata;
  std::vector<std::string> header;
  stan_csv_adaptation adaptation;
  Eigen::MatrixXd samples;
  stan_csv_timing timing;
};

/**
 * Reads from a Stan output csv file.
 */
class stan_csv_reader {
 public:
  stan_csv_reader() {}
  ~stan_csv_reader() {}

  static void read_metadata(std::istream& in, stan_csv_metadata& metadata) {
    std::stringstream ss;
    std::string line;

    if (in.peek() != '#')
      return;
    while (in.peek() == '#') {
      std::getline(in, line);
      ss << line << '\n';
    }
    ss.seekg(std::ios_base::beg);

    char comment;
    std::string lhs;

    std::string name;
    std::string value;

    while (ss.good()) {
      ss >> comment;
      std::getline(ss, lhs);

      size_t equal = lhs.find("=");
      if (equal != std::string::npos) {
        name = lhs.substr(0, equal);
        boost::trim(name);
        value = lhs.substr(equal + 1, lhs.size());
        boost::trim(value);
        boost::replace_first(value, " (Default)", "");
      } else {
        if (lhs.compare(" data") == 0) {
          ss >> comment;
          std::getline(ss, lhs);

          size_t equal = lhs.find("=");
          if (equal != std::string::npos) {
            name = lhs.substr(0, equal);
            boost::trim(name);
            value = lhs.substr(equal + 2, lhs.size());
            boost::replace_first(value, " (Default)", "");
          }

          if (name.compare("file") == 0)
            metadata.data = value;

          continue;
        }
      }

      if (name.compare("stan_version_major") == 0) {
        std::stringstream(value) >> metadata.stan_version_major;
      } else if (name.compare("stan_version_minor") == 0) {
        std::stringstream(value) >> metadata.stan_version_minor;
      } else if (name.compare("stan_version_patch") == 0) {
        std::stringstream(value) >> metadata.stan_version_patch;
      } else if (name.compare("model") == 0) {
        metadata.model = value;
      } else if (name.compare("num_samples") == 0) {
        std::stringstream(value) >> metadata.num_samples;
      } else if (name.compare("output_samples") == 0) {  // ADVI config name
        std::stringstream(value) >> metadata.num_samples;
      } else if (name.compare("num_warmup") == 0) {
        std::stringstream(value) >> metadata.num_warmup;
      } else if (name.compare("save_warmup") == 0) {
        // cmdstan args can be "true" and "false", was "1", "0"
        if (value.compare("true") == 0) {
          value = "1";
        }
        std::stringstream(value) >> metadata.save_warmup;
      } else if (name.compare("thin") == 0) {
        std::stringstream(value) >> metadata.thin;
      } else if (name.compare("id") == 0) {
        std::stringstream(value) >> metadata.chain_id;
      } else if (name.compare("init") == 0) {
        metadata.init = value;
        boost::trim(metadata.init);
      } else if (name.compare("seed") == 0) {
        std::stringstream(value) >> metadata.seed;
        metadata.random_seed = false;
      } else if (name.compare("append_samples") == 0) {
        std::stringstream(value) >> metadata.append_samples;
      } else if (name.compare("method") == 0) {
        metadata.method = value;
      } else if (name.compare("algorithm") == 0) {
        metadata.algorithm = value;
      } else if (name.compare("engine") == 0) {
        metadata.engine = value;
      } else if (name.compare("max_depth") == 0) {
        std::stringstream(value) >> metadata.max_depth;
      }
    }
  }  // read_metadata

  static bool read_header(std::istream& in, std::vector<std::string>& header,
                          bool prettify_name = true) {
    std::string line;

    if (!std::isalpha(in.peek()))
      return false;

    std::getline(in, line);
    std::stringstream ss(line);

    header.resize(std::count(line.begin(), line.end(), ',') + 1);
    int idx = 0;
    while (ss.good()) {
      std::string token;
      std::getline(ss, token, ',');
      boost::trim(token);

      if (prettify_name) {
        prettify_stan_csv_name(token);
      }
      header[idx++] = token;
    }
    return true;
  }

  static void read_adaptation(std::istream& in,
                              stan_csv_adaptation& adaptation) {
    std::stringstream ss;
    std::string line;
    int lines = 0;
    if (in.peek() != '#' || in.good() == false)
      return;
    while (in.peek() == '#') {
      std::getline(in, line);
      ss << line << std::endl;
      lines++;
    }
    ss.seekg(std::ios_base::beg);
    if (lines < 2)
      return;

    std::getline(ss, line);  // comment adaptation terminated

    // parse stepsize
    std::getline(ss, line, '=');  // stepsize
    boost::trim(line);
    ss >> adaptation.step_size;
    if (lines == 2)  // ADVI reports stepsize, no metric
      return;

    std::getline(ss, line);  // consume end of stepsize line
    std::getline(ss, line);  // comment elements of mass matrix
    std::getline(ss, line);  // diagonal metric or row 1 of dense metric

    int rows = lines - 3;
    int cols = std::count(line.begin(), line.end(), ',') + 1;
    if (cols == 1) {
      // model has no parameters
      return;
    }
    adaptation.metric.resize(rows, cols);
    char comment;  // Buffer for comment indicator, #

    // parse metric, row by row, element by element
    for (int row = 0; row < rows; row++) {
      std::stringstream line_ss;
      line_ss.str(line);
      line_ss >> comment;
      for (int col = 0; col < cols; col++) {
        std::string token;
        std::getline(line_ss, token, ',');
        boost::trim(token);
        std::stringstream(token) >> adaptation.metric(row, col);
      }
      std::getline(ss, line);
    }
  }

  static bool read_samples(std::istream& in, Eigen::MatrixXd& samples,
                           stan_csv_timing& timing) {
    std::stringstream ss;
    std::string line;

    int rows = 0;
    int cols = -1;

    if (in.peek() == '#' || in.good() == false)
      return false;  // need at least one data row

    while (in.good()) {
      bool comment_line = (in.peek() == '#');
      bool empty_line = (in.peek() == '\n');

      std::getline(in, line);
      if (empty_line)
        continue;
      if (!line.length())
        break;

      if (comment_line) {
        if (line.find("(Warm-up)") != std::string::npos) {
          int left = 17;
          int right = line.find(" seconds");
          double warmup;
          std::stringstream(line.substr(left, right - left)) >> warmup;
          timing.warmup += warmup;
        } else if (line.find("(Sampling)") != std::string::npos) {
          int left = 17;
          int right = line.find(" seconds");
          double sampling;
          std::stringstream(line.substr(left, right - left)) >> sampling;
          timing.sampling += sampling;
        }
      } else {
        ss << line << '\n';
        int current_cols = std::count(line.begin(), line.end(), ',') + 1;
        if (cols == -1) {
          cols = current_cols;
        } else if (cols != current_cols) {
          std::stringstream msg;
          msg << "Error: expected " << cols << " columns, but found "
              << current_cols << " instead for row " << rows + 1;
          throw std::invalid_argument(msg.str());
        }
        rows++;
      }

      in.peek();
    }

    ss.seekg(std::ios_base::beg);

    if (rows > 0) {
      samples.resize(rows, cols);
      for (int row = 0; row < rows; row++) {
        std::getline(ss, line);
        std::stringstream ls(line);
        for (int col = 0; col < cols; col++) {
          std::getline(ls, line, ',');
          boost::trim(line);
          try {
            samples(row, col) = static_cast<double>(std::stold(line));
            // If the value read is out of the range of representable values by
            // a long double
          } catch (const std::out_of_range& e) {
            samples(row, col) = std::numeric_limits<double>::quiet_NaN();
            // If no conversion could be performed, an invalid_argument
            // exception is thrown.
          } catch (const std::invalid_argument& e) {
            samples(row, col) = std::numeric_limits<double>::quiet_NaN();
          }
        }
      }
    }
    return true;
  }

  /**
   * Parses the file.
   *
   * Throws exception if contents can't be parsed into header + data rows.
   *
   * Emits warning message
   *
   * @param[in] in input stream to parse
   * @param[out] out output stream to send messages
   */
  static stan_csv parse(std::istream& in, std::ostream* out) {
    stan_csv data;
    std::string line;

    read_metadata(in, data.metadata);
    if (!read_header(in, data.header)) {
      throw std::invalid_argument("Error: no column names found in csv file");
    }

    // skip warmup draws, if any
    if (data.metadata.algorithm != "fixed_param" && data.metadata.num_warmup > 0
        && data.metadata.save_warmup) {
      while (in.peek() != '#') {
        std::getline(in, line);
      }
    }

    if (data.metadata.algorithm != "fixed_param") {
      read_adaptation(in, data.adaptation);
    }

    data.timing.warmup = 0;
    data.timing.sampling = 0;

    if (data.metadata.method == "variational") {
      std::getline(in, line);  // discard variational estimate
    }

    if (!read_samples(in, data.samples, data.timing)) {
      if (out)
        *out << "No draws found" << std::endl;
    }
    return data;
  }
};

}  // namespace io

}  // namespace stan

#endif
