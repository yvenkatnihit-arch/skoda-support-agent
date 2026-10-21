# Seed data: sources and caveats

All files were compiled on **2026-10-17** from public sources. Each row carries its own `source_url`
(and `as_of` where relevant), so the agent can cite where every figure came from. Prices change often:
re-check them before relying on a quote, and update `price_as_of` / `as_of` when you do.

Rows are validated on load by `app/data/schemas.py`. A malformed row stops startup with the file,
line and field. Run `python -m app.data.repository` after editing to check everything still loads.

| File | What it holds | Source |
|---|---|---|
| `models.json` | 6 models with `status` (`on_sale` / `upcoming`) | skoda-auto.co.in model pages |
| `variants.csv` | 41 variants: ex-showroom price (New Delhi), engine, power, torque, mileage, tank | Prices: CarDekho variant lists (Slavia Classic and Octavia RS from skoda-auto.co.in). Specs: skoda-auto.co.in, CarWale, Autocar India |
| `variant_features.csv` | Key features per variant (cumulative by trim) | Autocar India "which variant should you buy" guides; skoda-auto.co.in for Kodiaq and Octavia RS |
| `rto_rates.csv` | Road tax slabs for Telangana, Karnataka, Maharashtra, Delhi, with cess | motomotar.com state-wise table (rates as of March 2026) |
| `service_schedule.csv` | Service visits up to 5 years / 75,000 km with estimated cost | v3cars.com maintenance-cost pages |
| `dealers.csv` | 39 official showrooms and service centres in Hyderabad, Bengaluru, Mumbai/Thane, Delhi | dealers.skoda-auto.co.in (official locator; coordinates and hours from each outlet page) |
| `pincodes.csv` | 42 area pincodes with coordinates (offline fallback for geocoding) | OpenStreetMap Nominatim |
| `pricing_rules.json` | Insurance %, TCS, registration/HSRP, FASTag, handling, accessories | See `source_url` on each block |

## Known gaps and assumptions

- **No Skoda EV is on sale in India.** The Enyaq row is `status: upcoming`, and its price (₹50 lakh),
  battery (63 kWh) and range (500 km, international WLTP) are media estimates from CarWale. The agent
  must not give an on-road quote for it.
- **Superb** appears on skoda-auto.co.in but CarDekho lists it as discontinued, so it is left out.
- **Octavia RS** has no published ARAI mileage, so `mileage_kmpl` is blank and range can't be calculated.
- **Kodiaq (2nd gen)** fuel tank capacity isn't confirmed for the India model, so `fuel_tank_l` is blank.
- **Octavia RS and Enyaq** have no service schedule.
- **Kylaq** has no sourced accessory prices. Kushaq and Slavia have individual accessory prices.
- Kushaq **Sportline 1.5 DSG** is listed on the official site but had no price in the sources, so it is left out.
- **Insurance %, FASTag fee and handling charges are assumptions** (`is_assumption: true`). Replace them
  with insurer or dealer figures when you have them.
- **Delhi EV road tax** status is unclear: the old waiver lapsed in March 2026 and a new EV policy is still
  a draft. The file assumes 0% up to ₹30 lakh and 10% above.
- `variant_features.csv` is built up the trim ladder; since 2026-10-19 an upgrade replaces the item it supersedes
  (e.g. alloys replace steel wheels, a 10-inch screen replaces the 7-inch one), so each variant lists what it actually has.
- HTML entities (`&amp;`) in dealer addresses scraped from the locator were decoded on 2026-10-19; a test now guards against them.
- Dealer `services_offered` is inferred from the outlet type (showroom vs service centre), not from an official list.
- Ex-showroom prices are New Delhi figures and are used for every city. Actual prices can differ slightly by state.

## Updating the data

1. Edit the CSV/JSON file, keeping the same columns.
2. Update `source_url` and `as_of` / `price_as_of` on the rows you changed.
3. Run `python -m app.data.repository` and `pytest tests/test_repository.py`.
