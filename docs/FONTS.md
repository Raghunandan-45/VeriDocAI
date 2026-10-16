# Fonts

Font search done in Phase 0 (2026-10-16) on the local WSL2 Ubuntu machine, within the
15-minute cap. Sources checked: `fc-list`, matplotlib's bundled DejaVu
(`matplotlib/mpl-data/fonts/ttf/`), Debian/Ubuntu font packages and their
`/usr/share/doc/<pkg>/copyright` files. No pip font packages were installed (not on the
allowed dependency list). Glyph coverage was checked with fontTools (`getBestCmap`) and
every file listed below loads with `PIL.ImageFont.truetype`.

## Usable families

| Family | Package (version) | License | Basic Latin + `.,:/-#%()&@` | ₹ U+20B9 | € | £ | Proposed role |
|---|---|---|---|---|---|---|---|
| DejaVu Sans | fonts-dejavu-core 2.37 (also bundled with matplotlib) | Bitstream Vera license (DejaVu changes public domain) | complete | yes | yes | yes | train/val/test |
| DejaVu Serif | fonts-dejavu-core 2.37 | Bitstream Vera license | complete | yes | yes | yes | train/val/test |
| DejaVu Sans Mono | fonts-dejavu-core 2.37 | Bitstream Vera license | complete | yes | yes | yes | train/val/test |
| Liberation Sans | fonts-liberation 1.07.4 | GPL-2 with font exception (Red Hat) | complete | **no** | yes | yes | train/val/test |
| Liberation Serif | fonts-liberation 1.07.4 | GPL-2 with font exception | complete | **no** | yes | yes | train/val/test |
| Liberation Mono | fonts-liberation 1.07.4 | GPL-2 with font exception | complete | **no** | yes | yes | train/val/test |
| Lato | fonts-lato 2.0 | SIL OFL 1.1 | complete | yes | yes | yes | **held out** (test_unseen, stretch) |
| Ubuntu | fonts-ubuntu 0.83 | Ubuntu Font Licence 1.0 | complete | yes | yes | yes | **held out** (test_unseen, stretch) |

Every family above has Regular and Bold weights installed (bold is used for headers).

## Excluded

| Font | Reason |
|---|---|
| Liberation Sans Narrow | Condensed variant of Liberation Sans; not a distinct family. Could be added as a width variant later. |
| DejaVu Sans/Serif Condensed, DejaVu Sans Light | Variants of an already-used family. |
| DejaVu Math TeX Gyre, LyX `cm*`/`ms*`/`eufm`/`rsfs`/... | Math/symbol fonts; `cmr10` lacks € and em-dash. |
| Windows fonts under `/mnt/c/Windows/Fonts` (539 files) | Proprietary licenses, outside the project directory, and absent on Kaggle. Not used. |

## Limitations and decisions

- **Rupee sign:** Liberation 1.07.4 has no ₹ glyph. If amounts used ₹, only some fonts could
  render it, which would tie a glyph to a font family (a shortcut the model could learn).
  Decided (D2): all currency amounts are rendered as `Rs. 12,345.00` in every font.
- **Family counting for the unseen-font split:** held-out fonts are chosen at the
  *superfamily* level (all of Lato, all of Ubuntu) so that no variant of a held-out design
  leaks into training. Training then has 6 families from 2 superfamilies (DejaVu, Liberation);
  this satisfies "at least 3 training families" but the training design space is narrow.
  Decided (D3): the unseen-font test therefore means "unseen font **design**" and tests
  transfer to 2 new humanist sans designs only.
- **Reproducibility across machines (decided, D1):** Kaggle images are not guaranteed to ship
  Liberation, Lato or Ubuntu, and a different font file version changes rendered pixels and the
  dataset hash. The 16 TTF files actually used (Regular + Bold of the 8 families, 5.6 MB) are
  vendored in `assets/fonts/` with their license texts in `assets/fonts/licenses/`, and the
  generator loads fonts only from there (never from system font paths).
- Rendered pixels also depend on Pillow's bundled FreeType (2.14.3 with Pillow 12.3.0) and
  libjpeg-turbo; Pillow is pinned exactly in requirements.txt for this reason.

## Vendored files (`assets/fonts/`)

Copied from the Ubuntu packages listed above on 2026-10-16. sha256 from `sha256sum`.

| File | Family | Style | Role | sha256 |
|---|---|---|---|---|
| `DejaVuSans.ttf` | DejaVu Sans | Regular | train | `690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648` |
| `DejaVuSans-Bold.ttf` | DejaVu Sans | Bold | train | `d1c3ff99f1e1ce1827a33efd4dad81f40babda06bff9e43bd7591c86662a287b` |
| `DejaVuSerif.ttf` | DejaVu Serif | Regular | train | `3731a19f2b50d86d81dd1a1effcd81492ed1fa44f6ea904792dd98fdae9b3dfe` |
| `DejaVuSerif-Bold.ttf` | DejaVu Serif | Bold | train | `f81ce497906182460f9a4beb89bec7c1265af2182f8110c25821dfc88486942e` |
| `DejaVuSansMono.ttf` | DejaVu Sans Mono | Regular | train | `39c29931201f08dd89fdba4129c76288e6baeecf7c94fe4f6a757f2b50718b1b` |
| `DejaVuSansMono-Bold.ttf` | DejaVu Sans Mono | Bold | train | `f8aa70b5d4210d77624f5b5b5cf094fadf9fd019caf5b528fa5af42fc4ac43a5` |
| `LiberationSans-Regular.ttf` | Liberation Sans | Regular | train | `36efda825bcd7d1de846204c3adad51ced9d9b5cb7df571fb33edc21207e324a` |
| `LiberationSans-Bold.ttf` | Liberation Sans | Bold | train | `c2d334cafbd28522007cc16970db8db4a75256123594a43fe07f89cfceb5bacc` |
| `LiberationSerif-Regular.ttf` | Liberation Serif | Regular | train | `8d6951ea5fc4a9656df4802227292c7943186364929f00b97629e70daf228439` |
| `LiberationSerif-Bold.ttf` | Liberation Serif | Bold | train | `cfbe3d2edd46ae844b7eb95012f9274776801873e27b2230bbf751645f011888` |
| `LiberationMono-Regular.ttf` | Liberation Mono | Regular | train | `babd36837e644f08e6ba09b69feb770fbeba9e4611a619e5471fb8dbc132d42c` |
| `LiberationMono-Bold.ttf` | Liberation Mono | Bold | train | `410af69b6297b7cf6d20dd004004bd1bdb46460a60f9fef69b04169f4ff09660` |
| `Lato-Regular.ttf` | Lato | Regular | held out | `0ad460bd756454f8485609747b25c5644a54d307a65daabbb24c646c112ed541` |
| `Lato-Bold.ttf` | Lato | Bold | held out | `e0b1c4de011de8dd077cf21f97919fc31158eb290ee1176cbae06d500a6304bc` |
| `Ubuntu-R.ttf` | Ubuntu | Regular | held out | `52c1afa489ae7bfd893af6cdd9f1af258005703600449e70d338caabcff507e5` |
| `Ubuntu-B.ttf` | Ubuntu | Bold | held out | `c90e629e932f768909bdce1481ccd259d8aff1490a4bc1cf94d5413492317336` |

"train" = used for train/val/test rendering. "held out" = reserved for the stretch
`test_unseen` split only. License texts: `assets/fonts/licenses/{DejaVu,Liberation,Lato,Ubuntu}-LICENSE.txt`
(the Debian `copyright` files, which contain the full license terms).
