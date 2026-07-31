"""Economics Controller Tests"""

# pylint: disable=missing-function-docstring


def test_get_gross_domestic_product(recorder, economics_module):
    recorder.capture(economics_module.get_gross_domestic_product())
    recorder.capture(economics_module.get_gross_domestic_product(growth=True))
    recorder.capture(
        economics_module.get_gross_domestic_product(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        economics_module.get_gross_domestic_product(
            rolling=2, countries=["Netherlands", "Germany"]
        )
    )
    recorder.capture(
        economics_module.get_gross_domestic_product(
            trailing=2, countries=["Netherlands", "Germany"]
        )
    )
    recorder.capture(
        economics_module.get_gross_domestic_product(
            rolling=2, growth=True, countries=["Netherlands", "Germany"]
        )
    )


def test_get_gross_domestic_product_deflator(recorder, economics_module):
    recorder.capture(economics_module.get_gross_domestic_product_deflator())
    recorder.capture(economics_module.get_gross_domestic_product_deflator(growth=True))


def test_get_total_consumption(recorder, economics_module):
    recorder.capture(economics_module.get_total_consumption())
    recorder.capture(economics_module.get_total_consumption(growth=True))


def test_get_total_consumption_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_total_consumption_to_gdp_ratio())
    recorder.capture(economics_module.get_total_consumption_to_gdp_ratio(growth=True))


def test_get_investment(recorder, economics_module):
    recorder.capture(economics_module.get_investment())
    recorder.capture(economics_module.get_investment(growth=True))


def test_get_investment_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_investment_to_gdp_ratio())
    recorder.capture(economics_module.get_investment_to_gdp_ratio(growth=True))


def test_get_fixed_investment(recorder, economics_module):
    recorder.capture(economics_module.get_fixed_investment())
    recorder.capture(economics_module.get_fixed_investment(growth=True))


def test_get_fixed_investment_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_fixed_investment_to_gdp_ratio())
    recorder.capture(economics_module.get_fixed_investment_to_gdp_ratio(growth=True))


def test_get_exports(recorder, economics_module):
    recorder.capture(economics_module.get_exports())
    recorder.capture(economics_module.get_exports(growth=True))


def test_get_exports_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_exports_to_gdp_ratio())
    recorder.capture(economics_module.get_exports_to_gdp_ratio(growth=True))


def test_get_imports(recorder, economics_module):
    recorder.capture(economics_module.get_imports())
    recorder.capture(economics_module.get_imports(growth=True))


def test_get_imports_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_imports_to_gdp_ratio())
    recorder.capture(economics_module.get_imports_to_gdp_ratio(growth=True))


def test_get_current_account_balance(recorder, economics_module):
    recorder.capture(economics_module.get_current_account_balance())
    recorder.capture(economics_module.get_current_account_balance(growth=True))


def test_get_current_account_balance_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_current_account_balance_to_gdp_ratio())
    recorder.capture(
        economics_module.get_current_account_balance_to_gdp_ratio(growth=True)
    )


def test_get_government_debt(recorder, economics_module):
    recorder.capture(economics_module.get_government_debt())
    recorder.capture(economics_module.get_government_debt(growth=True))


def test_get_government_debt_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_government_debt_to_gdp_ratio())
    recorder.capture(economics_module.get_government_debt_to_gdp_ratio(growth=True))


def test_get_government_revenue(recorder, economics_module):
    recorder.capture(economics_module.get_government_revenue())
    recorder.capture(economics_module.get_government_revenue(growth=True))


def test_get_government_revenue_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_government_revenue_to_gdp_ratio())
    recorder.capture(economics_module.get_government_revenue_to_gdp_ratio(growth=True))


def test_get_government_tax_revenue(recorder, economics_module):
    recorder.capture(economics_module.get_government_tax_revenue())
    recorder.capture(economics_module.get_government_tax_revenue(growth=True))


def test_get_government_tax_revenue_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_government_tax_revenue_to_gdp_ratio())
    recorder.capture(
        economics_module.get_government_tax_revenue_to_gdp_ratio(growth=True)
    )


def test_get_government_expenditure(recorder, economics_module):
    recorder.capture(economics_module.get_government_expenditure())
    recorder.capture(economics_module.get_government_expenditure(growth=True))


def test_get_government_expenditure_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_government_expenditure_to_gdp_ratio())
    recorder.capture(
        economics_module.get_government_expenditure_to_gdp_ratio(growth=True)
    )


def test_get_government_deficit(recorder, economics_module):
    recorder.capture(economics_module.get_government_deficit())
    recorder.capture(economics_module.get_government_deficit(growth=True))


def test_get_government_deficit_to_gdp_ratio(recorder, economics_module):
    recorder.capture(economics_module.get_government_deficit_to_gdp_ratio())
    recorder.capture(economics_module.get_government_deficit_to_gdp_ratio(growth=True))
    recorder.capture(
        economics_module.get_government_deficit_to_gdp_ratio(
            countries=["Netherlands", "Germany"], rolling=2
        )
    )


def test_get_trust_in_government(recorder, economics_module):
    recorder.capture(economics_module.get_trust_in_government())
    recorder.capture(economics_module.get_trust_in_government(growth=True))


def test_get_consumer_price_index(recorder, economics_module):
    recorder.capture(economics_module.get_consumer_price_index())
    recorder.capture(economics_module.get_consumer_price_index(growth=True))


def test_get_inflation_rate(recorder, economics_module):
    recorder.capture(economics_module.get_inflation_rate())
    recorder.capture(economics_module.get_inflation_rate(growth=True))


def test_get_consumer_confidence_index(recorder, economics_module):
    recorder.capture(economics_module.get_consumer_confidence_index())
    recorder.capture(economics_module.get_consumer_confidence_index(growth=True))


def test_get_business_confidence_index(recorder, economics_module):
    recorder.capture(economics_module.get_business_confidence_index())
    recorder.capture(economics_module.get_business_confidence_index(growth=True))


def test_get_composite_leading_indicator(recorder, economics_module):
    recorder.capture(economics_module.get_composite_leading_indicator())
    recorder.capture(economics_module.get_composite_leading_indicator(growth=True))


def test_get_house_prices(recorder, economics_module):
    recorder.capture(economics_module.get_house_prices())
    recorder.capture(economics_module.get_house_prices(growth=True))
    recorder.capture(
        economics_module.get_house_prices(
            countries=["Netherlands", "Germany"], rolling=2
        )
    )


def test_get_rent_prices(recorder, economics_module):
    recorder.capture(economics_module.get_rent_prices())
    recorder.capture(economics_module.get_rent_prices(growth=True))


def test_get_share_prices(recorder, economics_module):
    recorder.capture(economics_module.get_share_prices())
    recorder.capture(economics_module.get_share_prices(growth=True))


def test_get_exchange_rates(recorder, economics_module):
    recorder.capture(economics_module.get_exchange_rates())
    recorder.capture(economics_module.get_exchange_rates(growth=True))


def test_get_money_supply(recorder, economics_module):
    recorder.capture(economics_module.get_money_supply())
    recorder.capture(economics_module.get_money_supply(growth=True))
    recorder.capture(
        economics_module.get_money_supply(
            measure="M2", countries=["Netherlands", "Germany"], rolling=2
        )
    )
    recorder.capture(
        economics_module.get_money_supply(
            measure="M2", countries=["Netherlands", "Germany"], trailing=2
        )
    )


def test_get_central_bank_policy_rate(recorder, economics_module):
    recorder.capture(economics_module.get_central_bank_policy_rate())
    recorder.capture(economics_module.get_central_bank_policy_rate(growth=True))


def test_get_short_term_interest_rate(recorder, economics_module):
    recorder.capture(economics_module.get_short_term_interest_rate())
    recorder.capture(economics_module.get_short_term_interest_rate(growth=True))


def test_get_long_term_interest_rate(recorder, economics_module):
    recorder.capture(economics_module.get_long_term_interest_rate())
    recorder.capture(economics_module.get_long_term_interest_rate(growth=True))


def test_get_renewable_energy(recorder, economics_module):
    recorder.capture(economics_module.get_renewable_energy())
    recorder.capture(economics_module.get_renewable_energy(growth=True))


def test_get_carbon_footprint(recorder, economics_module):
    recorder.capture(economics_module.get_carbon_footprint())
    recorder.capture(economics_module.get_carbon_footprint(growth=True))


def test_get_unemployment_rate(recorder, economics_module):
    recorder.capture(economics_module.get_unemployment_rate())
    recorder.capture(economics_module.get_unemployment_rate(growth=True))


def test_get_labour_productivity(recorder, economics_module):
    recorder.capture(economics_module.get_labour_productivity())
    recorder.capture(economics_module.get_labour_productivity(growth=True))


def test_get_income_inequality(recorder, economics_module):
    recorder.capture(economics_module.get_income_inequality())
    recorder.capture(economics_module.get_income_inequality(growth=True))
    recorder.capture(
        economics_module.get_income_inequality(countries="United States", rolling=2)
    )


def test_get_population_statistics(recorder, economics_module):
    recorder.capture(economics_module.get_population_statistics())
    recorder.capture(economics_module.get_population_statistics(growth=True))


def test_get_poverty_rate(recorder, economics_module):
    recorder.capture(economics_module.get_poverty_rate())
    recorder.capture(economics_module.get_poverty_rate(growth=True))


def test_get_real_gross_domestic_product_usd(recorder, economics_module):
    recorder.capture(economics_module.get_real_gross_domestic_product_usd())
    recorder.capture(economics_module.get_real_gross_domestic_product_usd(growth=True))


def test_get_real_gross_domestic_product_per_capita(recorder, economics_module):
    recorder.capture(economics_module.get_real_gross_domestic_product_per_capita())
    recorder.capture(
        economics_module.get_real_gross_domestic_product_per_capita(growth=True)
    )


def test_get_output_gap(recorder, economics_module):
    recorder.capture(economics_module.get_output_gap())
    recorder.capture(economics_module.get_output_gap(growth=True))
    recorder.capture(
        economics_module.get_output_gap(countries=["United States", "Germany", "Japan"])
    )


def test_get_trade_balance(recorder, economics_module):
    recorder.capture(economics_module.get_trade_balance())
    recorder.capture(economics_module.get_trade_balance(growth=True))
    recorder.capture(
        economics_module.get_trade_balance(countries=["United States", "China"])
    )


def test_get_real_effective_exchange_rate(recorder, economics_module):
    recorder.capture(economics_module.get_real_effective_exchange_rate())
    recorder.capture(economics_module.get_real_effective_exchange_rate(growth=True))


def test_get_producer_price_index(recorder, economics_module):
    recorder.capture(economics_module.get_producer_price_index())
    recorder.capture(economics_module.get_producer_price_index(growth=True))
    recorder.capture(
        economics_module.get_producer_price_index(
            countries=["United States", "Germany"], period="yearly"
        )
    )


def test_get_real_interest_rate(recorder, economics_module):
    recorder.capture(economics_module.get_real_interest_rate())
    recorder.capture(economics_module.get_real_interest_rate(growth=True))
    recorder.capture(
        economics_module.get_real_interest_rate(
            countries=["United States", "Germany", "Japan"], rate_type="short_term"
        )
    )
    recorder.capture(
        economics_module.get_real_interest_rate(
            countries=["United States", "Germany"], gmdb_source=False
        )
    )


def test_get_misery_index(recorder, economics_module):
    recorder.capture(economics_module.get_misery_index())
    recorder.capture(economics_module.get_misery_index(growth=True))
    recorder.capture(
        economics_module.get_misery_index(
            countries=["United States", "Germany"], gmdb_source=False
        )
    )


def test_get_yield_curve_slope(recorder, economics_module):
    recorder.capture(economics_module.get_yield_curve_slope())
    recorder.capture(economics_module.get_yield_curve_slope(growth=True))
    recorder.capture(
        economics_module.get_yield_curve_slope(
            countries=["United States", "Germany"], gmdb_source=False
        )
    )


def test_get_sovereign_debt_crisis(recorder, economics_module):
    recorder.capture(economics_module.get_sovereign_debt_crisis())
    recorder.capture(economics_module.get_sovereign_debt_crisis(countries="Argentina"))


def test_get_currency_crisis(recorder, economics_module):
    recorder.capture(economics_module.get_currency_crisis())
    recorder.capture(economics_module.get_currency_crisis(countries="Turkey"))


def test_get_banking_crisis(recorder, economics_module):
    recorder.capture(economics_module.get_banking_crisis())
    recorder.capture(
        economics_module.get_banking_crisis(
            countries=["United States", "United Kingdom"]
        )
    )
