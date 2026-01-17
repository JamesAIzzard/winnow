from __future__ import annotations


from winnow.parser.base import Parser, ParserError


class SimpleParser(Parser[str]):
    """Test parser that returns the stripped response."""

    def parse(self, response: str) -> str:
        """Return stripped response."""
        return response.strip()


class TestParserDeclineDetection:
    def test_returns_none_for_unknown_keyword(self) -> None:
        """Verify parser returns None when response contains UNKNOWN."""
        parser = SimpleParser()

        result = parser(response="UNKNOWN")

        assert result is None

    def test_returns_none_for_insufficient_data_keyword(self) -> None:
        """Verify parser returns None when response contains INSUFFICIENT_DATA."""
        parser = SimpleParser()

        result = parser(response="INSUFFICIENT_DATA")

        assert result is None

    def test_returns_none_for_mixed_case_keyword(self) -> None:
        """Verify decline detection is case-insensitive."""
        parser = SimpleParser()

        result = parser(response="Unknown")

        assert result is None

    def test_returns_none_for_keyword_in_sentence(self) -> None:
        """Verify decline detection finds keyword within text."""
        parser = SimpleParser()

        result = parser(response="I am UNKNOWN about this")

        assert result is None

    def test_parses_normal_response(self) -> None:
        """Verify parser returns parsed value for normal responses."""
        parser = SimpleParser()

        result = parser(response="  hello world  ")

        assert result == "hello world"


class TestParserCustomDeclineKeywords:
    def test_custom_decline_keywords(self) -> None:
        """Verify custom decline keywords are respected."""

        class CustomParser(Parser[str]):
            decline_keywords = frozenset({"SKIP", "PASS"})

            def parse(self, response: str) -> str:
                return response.strip()

        parser = CustomParser()

        assert parser(response="SKIP") is None
        assert parser(response="PASS") is None
        assert parser(response="UNKNOWN") == "UNKNOWN"


class TestParserError:
    def test_parser_error_is_parse_failed_error(self) -> None:
        """Verify ParserError is an alias for ParseFailedError."""
        from winnow.exceptions import ParseFailedError

        error = ParserError(response="test", reason="failed")

        assert isinstance(error, ParseFailedError)
        assert error.response == "test"
        assert error.reason == "failed"
