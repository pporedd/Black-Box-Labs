"""Tests for the VLM evaluator answer parser."""

from evaluator.vlm import MoondreamEvaluator


class TestAnswerParsing:
    """Tests for MoondreamEvaluator._parse_answer()."""

    def test_plain_digit(self):
        assert MoondreamEvaluator._parse_answer("3") == "3"

    def test_padded_digit(self):
        assert MoondreamEvaluator._parse_answer("  3  ") == "3"

    def test_digit_in_sentence(self):
        assert MoondreamEvaluator._parse_answer("There are 5 circles") == "5"

    def test_word_number(self):
        assert MoondreamEvaluator._parse_answer("three") == "3"

    def test_capitalized_word(self):
        assert MoondreamEvaluator._parse_answer("Three items") == "3"

    def test_word_in_sentence(self):
        assert MoondreamEvaluator._parse_answer("I see seven red circles") == "7"

    def test_zero(self):
        assert MoondreamEvaluator._parse_answer("zero") == "0"
        assert MoondreamEvaluator._parse_answer("0") == "0"

    def test_double_digit(self):
        assert MoondreamEvaluator._parse_answer("12") == "12"
        assert MoondreamEvaluator._parse_answer("twelve") == "12"


class TestComparison:
    """Tests for MoondreamEvaluator._compare()."""

    def test_exact_match(self):
        assert MoondreamEvaluator._compare("5", "5") is True

    def test_whitespace_tolerance(self):
        assert MoondreamEvaluator._compare(" 5 ", "5") is True

    def test_mismatch(self):
        assert MoondreamEvaluator._compare("3", "5") is False
