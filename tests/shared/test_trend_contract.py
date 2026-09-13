import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.contracts.trend_contract import TrendAdvisorRequest


@pytest.mark.parametrize("domain", ["Biotech", "cs_ai_ml", "OTHER"])
def test_unlisted_domain_is_rejected(domain):
    with pytest.raises(ValidationError):
        TrendAdvisorRequest(domain=domain)


def test_other_without_domain_other_name_is_rejected():
    with pytest.raises(ValidationError):
        TrendAdvisorRequest(domain="Other")


def test_other_with_domain_other_name_is_accepted():
    req = TrendAdvisorRequest(domain="Other", domain_other_name="Biotech")
    assert req.domain_label == "Biotech"


def test_fixed_domain_is_accepted():
    req = TrendAdvisorRequest(domain="MECHANICAL")
    assert req.domain_label == "MECHANICAL"
