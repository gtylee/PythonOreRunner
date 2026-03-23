from types import SimpleNamespace

from pythonore.domain.dataclasses import GenericProduct
from pythonore.mapping.mapper import _currency_config_xml, _filtered_loaded_portfolio_xml


def test_currency_config_xml_includes_base_and_trade_currencies():
    snapshot = SimpleNamespace(
        config=SimpleNamespace(base_currency="USD"),
        portfolio=SimpleNamespace(
            trades=(
                SimpleNamespace(product=SimpleNamespace(ccy="EUR")),
                SimpleNamespace(product=SimpleNamespace(ccy="USD")),
            )
        ),
    )

    xml = _currency_config_xml(snapshot)

    assert "<CurrencyConfig>" in xml
    assert "<ISOCode>USD</ISOCode>" in xml
    assert "<ISOCode>EUR</ISOCode>" in xml


def test_currency_config_xml_includes_currencies_from_generic_trade_payload():
    snapshot = SimpleNamespace(
        config=SimpleNamespace(base_currency="USD"),
        portfolio=SimpleNamespace(
            trades=(
                SimpleNamespace(
                    product=GenericProduct(
                        payload={
                            "xml": """
<CapFloorData>
  <LegData>
    <Currency>USD</Currency>
  </LegData>
</CapFloorData>
""".strip()
                        }
                    )
                ),
            )
        ),
    )

    xml = _currency_config_xml(snapshot)

    assert "<ISOCode>USD</ISOCode>" in xml


def test_filtered_loaded_portfolio_xml_preserves_original_trade_xml_for_subset():
    snapshot = SimpleNamespace(
        portfolio=SimpleNamespace(
            trades=(
                SimpleNamespace(trade_id="T2"),
            )
        )
    )
    xml_buffers = {
        "portfolio_example.xml": """
<Portfolio>
  <Trade id="T1"><TradeType>Swap</TradeType></Trade>
  <Trade id="T2"><TradeType>Swaption</TradeType><SwaptionData/></Trade>
</Portfolio>
""".strip()
    }

    filtered = _filtered_loaded_portfolio_xml(snapshot, xml_buffers)

    assert filtered is not None
    assert 'id="T2"' in filtered
    assert 'id="T1"' not in filtered
    assert "<SwaptionData" in filtered
