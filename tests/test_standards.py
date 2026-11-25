"""The four example markings from http://elsie.ua/eng/mark.html, as the
reader would deliver them, plus rule violations."""

import datetime
import unittest

from app.marking import standards as s

TODAY = datetime.date(2026, 10, 5)


def pilkington(**kw):
    args = dict(lines=["DOT682M3240AS2", "43R - 001582", "E 000199"],
                manufacturer="PILKINGTON", e_mark="E1", ccc_mark=True,
                date_code=["3..."], today=TODAY)
    args.update(kw)
    return s.parse(**args)


def sekurit():
    return s.parse(["SAINT-GOBAIN", "43 R - 00346", "DOT503 AS2 M381"],
                   manufacturer="SEKURIT", e_mark="E17", date_code=["...2"], today=TODAY)


def fuyao(**kw):
    args = dict(lines=["LAMINATED", "// M868 AS1", "DOT - 459", "43R-001298"],
                manufacturer="FUYAO", e_mark="E2", date_code=["9."], today=TODAY)
    args.update(kw)
    return s.parse(**args)


def xyg():
    return s.parse(["LAMINATED", "SAFETY GLASS", "DOT-563 AS-1 M-668", "43 R -001532", "E000039"],
                   manufacturer="XYG", e_mark="E1", ccc_mark=True, glass_class="//",
                   date_code=["9", "1 2 8", "10 1 2 4"], today=TODAY)


class ParseExamples(unittest.TestCase):
    def test_pilkington(self):
        r = pilkington()
        self.assertEqual(r.profile, ["ECE", "DOT", "CCC"])
        self.assertEqual((r.e_country, r.ece_approval), (1, "43R-001582"))
        self.assertEqual((r.dot_code, r.dot_model, r.as_class), ("682", "M3240", "AS2"))
        self.assertEqual(r.ccc_number, "E000199")
        self.assertEqual(str(r.date_code), "2023-09")  # page rule: 3 dots after = 3 + 6
        self.assertEqual(s.validate(r, today=TODAY), [])

    def test_sekurit(self):
        r = sekurit()
        self.assertEqual(r.profile, ["ECE", "DOT"])
        self.assertEqual((r.e_country, r.ece_approval), (17, "43R-00346"))
        self.assertEqual((r.dot_code, r.dot_model, r.as_class), ("503", "M381", "AS2"))
        self.assertEqual(str(r.date_code), "2022-03")
        self.assertEqual(r.unparsed, ["SAINT-GOBAIN"])
        self.assertEqual(s.validate(r, today=TODAY), [])

    def test_fuyao(self):
        r = fuyao()
        self.assertEqual(r.describe(), "FUYAO | ECE+DOT | LAMINATED")
        self.assertEqual(r.glass_class, "II")
        self.assertEqual((r.dot_code, r.dot_model, r.as_class), ("459", "M868", "AS1"))
        self.assertEqual(r.ece_approval, "43R-001298")
        self.assertEqual(str(r.date_code), "2019-07")
        self.assertEqual(s.validate(r, today=TODAY), [])

    def test_xyg_pyramid_date(self):
        r = xyg()
        self.assertEqual(r.profile, ["ECE", "DOT", "CCC"])
        self.assertEqual((r.glass_type, r.glass_class), ("LAMINATED", "II"))
        self.assertEqual((r.dot_code, r.dot_model, r.as_class), ("563", "M668", "AS1"))
        self.assertEqual(r.ccc_number, "E000039")
        self.assertEqual(str(r.date_code), "2019-11-17")
        self.assertEqual(s.validate(r, today=TODAY), [])

    def test_ocr_confusions_in_digit_fields(self):
        r = s.parse(["43R-OO1582", "DOTG82"], e_mark="EI7", today=TODAY)
        self.assertEqual(r.ece_approval, "43R-001582")
        self.assertEqual(r.e_country, 17)

    def test_misread_glass_type_word(self):
        self.assertEqual(s.parse(["LAMlNATED"]).glass_type, "LAMINATED")
        self.assertEqual(s.parse(["TEMPRED"]).glass_type, "TEMPERED")


class DateCodes(unittest.TestCase):
    def test_industry_rule(self):
        self.assertEqual(str(s.decode_date_code(["9."], "industry", TODAY)), "2019-12")
        self.assertEqual(str(s.decode_date_code(["......5"], "industry", TODAY)), "2025-01")

    def test_year_resolves_to_recent_past(self):
        self.assertEqual(s.decode_date_code([".6"], today=TODAY).year, 2026)
        self.assertEqual(s.decode_date_code([".7"], today=TODAY).year, 2027)
        self.assertEqual(s.decode_date_code([".8"], today=TODAY).year, 2018)

    def test_invalid(self):
        for rows in (["3"], [".3."], [".......3"], ["X.."], ["9", "13", "1"]):
            with self.assertRaises(ValueError, msg=rows):
                s.decode_date_code(rows, today=TODAY)


def codes(issues):
    return sorted(i.code for i in issues)


class RuleViolations(unittest.TestCase):
    def test_bad_country(self):
        self.assertIn("ECE_COUNTRY_INVALID", codes(s.validate(pilkington(e_mark="E99"), today=TODAY)))

    def test_short_approval(self):
        r = pilkington(lines=["DOT682M3240AS2", "43R-0015", "E 000199"])
        self.assertEqual(codes(s.validate(r, today=TODAY)), ["ECE_APPROVAL_FORMAT"])

    def test_bad_as_class(self):
        r = pilkington(lines=["DOT682 M3240 AS7", "43R-001582", "E 000199"])
        self.assertEqual(codes(s.validate(r, today=TODAY)), ["AS_CLASS_INVALID"])

    def test_missing_mandatory(self):
        r = s.parse(["DOT682"], today=TODAY)
        self.assertEqual(codes(s.validate(r, today=TODAY)), [
            "AS_CLASS_MISSING", "DOT_MODEL_MISSING", "ECE_APPROVAL_MISSING",
            "ECE_MARK_MISSING", "MANUFACTURER_MISSING"])

    def test_ccc_logo_without_number(self):
        r = pilkington(lines=["DOT682M3240AS2", "43R-001582"])
        self.assertEqual(codes(s.validate(r, today=TODAY)), ["CCC_NUMBER_MISSING"])

    def test_tempered_windshield(self):
        r = fuyao(lines=["TEMPERED", "M868 AS1", "DOT-459", "43R-001298"])
        self.assertEqual(codes(s.validate(r, today=TODAY)), ["AS1_NOT_LAMINATED"])

    def test_future_date(self):
        r = pilkington(date_code=["7.."])  # 2027-08
        self.assertEqual(codes(s.validate(r, today=TODAY)), ["DATE_IN_FUTURE"])

    def test_bad_date_code(self):
        r = pilkington(date_code=[".3."])
        self.assertEqual(codes(s.validate(r, today=TODAY)), ["DATE_CODE_INVALID"])


class MasterComparison(unittest.TestCase):
    MASTER = {"manufacturer": "PILKINGTON", "profile": ["ECE", "DOT", "CCC"],
              "e_country": 1, "ece_approval": "43R-001582", "dot_code": "682",
              "dot_model": "M3240", "as_class": "AS2", "ccc_number": "E000199",
              "date_max_age_days": 365 * 5}

    def test_match(self):
        self.assertEqual(s.validate(pilkington(), self.MASTER, today=TODAY), [])

    def test_wrong_approval_and_missing_ccc(self):
        r = pilkington(lines=["DOT682M3240AS2", "43R-001583"], ccc_mark=False)
        self.assertEqual(codes(s.validate(r, self.MASTER, today=TODAY)),
                         ["CCC_NUMBER_MISMATCH", "ECE_APPROVAL_MISMATCH", "PROFILE_MISMATCH"])

    def test_other_brand(self):
        self.assertEqual(codes(s.validate(sekurit(), {"manufacturer": "PILKINGTON"}, today=TODAY)),
                         ["MANUFACTURER_MISMATCH"])

    def test_old_stock(self):
        master = dict(self.MASTER, date_max_age_days=365)
        self.assertEqual(codes(s.validate(pilkington(), master, today=TODAY)), ["DATE_TOO_OLD"])


if __name__ == "__main__":
    unittest.main()
