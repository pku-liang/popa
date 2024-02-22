/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the BSD-2-Clause Plus Patent License (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSDplusPatent
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: BSD-2-Clause-Patent
*******************************************************************************/
#ifndef T2S_STRUCT_TYPE_H
#define T2S_STRUCT_TYPE_H

/** \file
 *
 * Provide a mechanism to create struct types in the IR.
 *
 */

#include "../../Halide/src/IREquality.h"
#include "../../Halide/src/IR.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::pair;
using std::tuple;

class GeneratedStructType {
private:
    bool struct_type_exists(const vector<Type> &field_types, size_t &index) {
        for (size_t i = 0; i < structs.size(); i++) {
            if (structs[i].second.size() != field_types.size()) {
                continue;
            }
            bool equal = true;
            for (size_t j = 0; j < field_types.size(); j++) {
                if (structs[i].second[j] != field_types[j]) {
                    equal = false;
                    break;
                }
            }
            if (equal) {
                index = i;
                return true;
            }
        }
        return false;
    }

public:
    // Non-standard vectors: basic type is standard, but #lanes is not. For example: float9 is not
    // a standard OpenCL type, and here we can create it.
    static vector<pair<Type, int>> vectors; // Each entry: basic type, lanes.
                                                          // For example, for float9: (float, 9)

    // Structs, composed of multiple fields.
    static vector<pair<string, vector<Type>>> structs; // Each entry: struct name, field types
    static halide_handle_cplusplus_type dummy;         // A dummy handle named "CGS" (compiler_generated_struct)

    static bool nonstandard_vector_type_exists(const Type &type) {
        Type basic_type = type.with_lanes(1);
        int lanes = type.lanes();
        for (auto v : vectors) {
            if (v.first == basic_type && v.second == lanes) {
                return true;
            }
        }
        return false;
    }

    static void record_nonstandard_vector_type(const Type & type) {
        if (nonstandard_vector_type_exists(type)) {
            return;
        }
        Type basic_type = type.with_lanes(1);
        int lanes = type.lanes();
        pair<Type, int> basic_type_lanes = pair<Type, int>(basic_type, lanes);
        vectors.push_back(basic_type_lanes);
    }

    GeneratedStructType(const vector<Type> &field_types, size_t &index) {
        if (struct_type_exists(field_types, index)) {
            return;
        }
        index = structs.size();
        internal_assert(index <= 0xFF) << "Too many structs generated.";
        string struct_name = unique_name("cgs");
        structs.push_back(pair<string, vector<Type>>(struct_name, field_types));
    }
};

class GeneratedArrayType {

    bool array_type_exists(Type t, const Region &ranges, size_t &index) {
        for (size_t i = 0; i < arrays.size(); i++) {
            bool found = true;
            auto cur_type = std::get<1>(arrays[i]);
            auto cur_ranges = std::get<2>(arrays[i]);
            if (t != cur_type || ranges.size() != cur_ranges.size())
                continue;
            for (size_t j = 0; j < ranges.size(); j++) {
                if (!equal(ranges[j].min, cur_ranges[j].min)
                    || !equal(ranges[j].extent, cur_ranges[j].extent)) found = false;
            }
            if (found) {
                index = i;
                return true;
            }
        }
        return false;
    }

public:
    static vector<tuple<string, Type, Region>> arrays; // Each entry: array name, base type, and bounds
    static halide_handle_cplusplus_type dummy;         // A dummy handle named "CGA" (compiler_generated_array)

    GeneratedArrayType(const Region &ranges, Type t, size_t &index) {
        if (array_type_exists(t, ranges, index)) {
            return;
        }
        index = arrays.size();
        internal_assert(index <= 0xFF) << "Too many arrays generated.";
        string array_name = unique_name("cga");
        arrays.push_back(tuple<string, Type, Region>(array_name, t, ranges));
    }
};


/* Given the types of the fields, generate a struct wrapped up as a Halide type so that it can be
 * used in the Halide IR. */
Type generate_struct(const vector<Type> &field_types);
Type generate_array(Type t, const vector<Range> &ranges);

}// Internal
} // Halide

#endif
