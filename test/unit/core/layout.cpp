/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("layout tags expose stable ranks and fastest axis", "[core][layout]")
{
    STATIC_CHECK(alpakaNN::layout::rank<alpakaNN::layout::TH>() == 2u);
    STATIC_CHECK(alpakaNN::layout::rank<alpakaNN::layout::BTH>() == 3u);
    STATIC_CHECK(alpakaNN::layout::rank<alpakaNN::layout::BTHD>() == 4u);
    STATIC_CHECK(alpakaNN::layout::rank<alpakaNN::layout::BHTD>() == 4u);
    STATIC_CHECK(alpakaNN::layout::rank<alpakaNN::layout::LBHTD>() == 5u);
    STATIC_CHECK(alpakaNN::layout::fastestAxis<alpakaNN::layout::LBHTD>() == 4u);
}
