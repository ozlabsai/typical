# Helipad demo

An inspection desk that replays recorded landing-zone inspections (data contract: [DATA.md](DATA.md)).

```bash
npm install
npm run dev                         # fixtures/ (synthetic S2, S4, O7)
HELIPAD_DEMO_DATA=/path/to/demo npm run dev   # recorded data, e.g. $HELIPAD_PRIVATE/demo
npm test                            # vitest: i18n coverage, data parsing, overlay geometry
npm run build
npm run fixtures                    # regenerate fixtures/ (synthetic records and placeholder images)
```
