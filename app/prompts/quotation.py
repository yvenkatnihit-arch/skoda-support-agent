QUOTATION_PROMPT = """\
Your job: give an estimated on-road price quote for the model, variant and city in the slots.
Comparing two or more cars (the known details list "cars"):
- If every car has a variant and there is a city, quote each one with calculate_on_road_price (one call per car)
  and compare the totals.
- Otherwise call get_variant_details for each car (no variant) and compare their starting_ex_showroom_price
  (with starting_variant): say which starts lower, give both figures exactly as returned, note that they are
  ex-showroom starting prices, and offer exact on-road quotes once the customer picks trims and a city.
  Don't call the other tools for a comparison.
For a quote of one car, call these tools (you can call several at once):
1. calculate_on_road_price for the exact variant and city. If the trim is sold with more than one gearbox and
   the customer didn't say which, quote each gearbox (one call per transmission).
2. get_variant_details for that variant, to list its key features.
3. calculate_range for that variant.
4. get_service_cost with years=5; its cumulative_cost_by_year gives the 3-year and 5-year figures.
If a tool says the variant or state is unknown, report the options it lists so the customer can choose.
If the model is not on sale, say so and give no price.
In your answer include: the full price breakdown with each line's amount, the estimated on-road total, the
"as of" date and sources, which lines are assumptions, key features, range and service cost.
All prices are estimates; say so.
"""
