"""A library of racing circuits to train and evaluate on.

Every layout here is a *schematic reconstruction* of a real circuit: the
sequence of corners, their radii, their direction and the length of the
straights between them are drawn from the published geometry, so the driving
problem is the real one -- a 1.1 km straight into a 28 m radius chicane at
Monza, a 14 m radius hairpin at the top of the hill at Brands Hatch. Lap
lengths land within a percent of the published figures, but the plan view is
close rather than survey-accurate: the numbers were fitted to close the loop
exactly, which trades a few metres of straight for the guarantee that the
circuit is a valid closed track. Treat these as circuits *in the character of*
their inspiration, not as substitutes for surveyed track data.

Three deliberate departures:

* Width is uniform per circuit, because `Track` models a constant width. The
  value used is representative of the real circuit (9 m for a street circuit,
  15 m for the Indianapolis oval).
* Figure-of-eight circuits (Suzuka) are absent. Their plan view crosses itself,
  and `Track` builds the drivable surface as an annulus -- an outer ring with
  the inner boundary as a hole -- which a self-crossing centerline cannot form.
* `monaco_street` is an original layout rather than a reconstruction. Monaco's
  real plan runs back alongside itself along the harbour, and every attempt at
  reconstructing it put the swimming pool section within 7 m of the start
  straight: at 9 m wide the two boundaries cross, which merges the outbound and
  return legs into one drivable surface and lets a lap be cut short. What is
  kept is the driving problem -- 9 m of width, seventeen slow corners, a 10 m
  radius hairpin and a chicane, over 3.3 km.

The `tags` on each layout are the handle for curriculum work: `easy` circuits
are fast and open, `hard` ones are narrow with tight corners.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.track import Track, resolution_for_arc_step, save_track_json
from utils.track_builder import Corner, Layout, Straight, build_centerline

# =============================================================================
# Ovals and short circuits -- the low end of the difficulty range
# =============================================================================

INDIANAPOLIS_OVAL = Layout(
    name="indianapolis_oval",
    width=15.0,
    inspiration="Indianapolis Motor Speedway (oval), anticlockwise",
    reference_length=4023.0,
    tags=("oval", "high-speed", "easy"),
    segments=(
        Straight(1006.0, "front straight"),
        Corner(256.0, 90.0, "Turn 1"),
        Straight(201.0, "short chute 1-2"),
        Corner(256.0, 90.0, "Turn 2"),
        Straight(1006.0, "back straight"),
        Corner(256.0, 90.0, "Turn 3"),
        Straight(201.0, "short chute 3-4"),
        Corner(256.0, 90.0, "Turn 4"),
    ),
)

BRANDS_HATCH_INDY = Layout(
    name="brands_hatch_indy",
    width=11.0,
    inspiration="Brands Hatch Indy circuit",
    reference_length=1929.0,
    tags=("short", "hairpin", "medium"),
    segments=(
        Straight(105.2, "start/finish"),
        Corner(58.0, -46.6, "Paddock Hill Bend"),
        Straight(365.6, "Hailwood Hill"),
        Corner(627.0, 23.9, "Hailwood Hill (kink)"),
        Corner(14.0, -160.2, "Druids"),
        Straight(300.3),
        Corner(35.8, -30.1, "Graham Hill Bend"),
        Straight(470.5, "Cooper Straight"),
        Corner(82.8, -36.7, "Surtees"),
        Straight(52.3),
        Corner(53.5, -49.9, "McLaren"),
        Corner(80.5, -35.8, "Clearways"),
        Corner(143.2, -24.6, "Clark Curve"),
        Straight(57.2),
    ),
)

# =============================================================================
# Permanent circuits
# =============================================================================

RED_BULL_RING = Layout(
    name="red_bull_ring",
    width=13.0,
    inspiration="Red Bull Ring, Spielberg",
    reference_length=4318.0,
    tags=("permanent", "high-speed", "easy"),
    segments=(
        Straight(604.1, "start/finish"),
        Corner(50.0, -89.9, "Niki Lauda Kurve"),
        Straight(773.7, "climb to Remus"),
        Corner(399.8, 41.3, "climb (kink)"),
        Corner(28.0, -109.9, "Remus"),
        Straight(523.2, "to Schlossgold"),
        Corner(55.0, -65.8, "Schlossgold"),
        Straight(186.8),
        Corner(300.1, 25.6, "Kurve 4"),
        Corner(80.0, -56.8, "Wurth Kurve"),
        Straight(315.2),
        Corner(60.0, 68.8, "Kurve 6"),
        Straight(121.0),
        Corner(45.0, -78.4, "Rauch Kurve"),
        Straight(450.5),
        Corner(70.0, -61.1, "Jochen Rindt Kurve"),
        Straight(212.6),
        Corner(120.0, -33.9, "Red Bull Mobile Kurve"),
        Straight(154.8),
    ),
)

MONZA = Layout(
    name="monza",
    width=13.0,
    inspiration="Autodromo Nazionale di Monza",
    reference_length=5793.0,
    tags=("permanent", "high-speed", "chicane", "medium"),
    segments=(
        Straight(1110.4, "Rettifilo Tribune"),
        Corner(28.0, -75.0, "Variante del Rettifilo (1)"),
        Corner(28.0, 70.0, "Variante del Rettifilo (2)"),
        Straight(138.3, "to Curva Biassono"),
        Corner(320.0, -70.0, "Curva Grande"),
        Straight(696.0, "to Variante della Roggia"),
        Corner(38.0, 64.9, "Variante della Roggia (1)"),
        Corner(38.0, -65.0, "Variante della Roggia (2)"),
        Straight(385.4, "to Lesmo"),
        Corner(58.1, -83.8, "Curva di Lesmo 1"),
        Straight(154.2),
        Corner(48.0, -79.1, "Curva di Lesmo 2"),
        Straight(554.5, "Serraglio"),
        Corner(90.0, 54.7, "Variante Ascari (1)"),
        Corner(58.0, -70.8, "Variante Ascari (2)"),
        Corner(100.0, 49.5, "Variante Ascari (3)"),
        Straight(1404.7, "Rettifilo Centro"),
        Corner(150.0, -155.2, "Curva Parabolica"),
    ),
)

SPA_FRANCORCHAMPS = Layout(
    name="spa_francorchamps",
    width=13.0,
    inspiration="Circuit de Spa-Francorchamps",
    reference_length=7004.0,
    tags=("permanent", "high-speed", "long", "medium"),
    segments=(
        Straight(576.9, "Ligne Droite des Stands"),
        Corner(22.0, -175.2, "La Source"),
        Straight(181.3, "descent to Eau Rouge"),
        Corner(150.0, 25.0, "Eau Rouge"),
        Corner(90.0, -45.0, "Raidillon (1)"),
        Corner(200.0, 25.0, "Raidillon (2)"),
        Straight(869.8, "Kemmel"),
        Corner(1489.4, -15.0, "Kemmel (sweep)"),
        Corner(45.0, -94.2, "Les Combes (1)"),
        Corner(50.0, 44.0, "Les Combes (2)"),
        Straight(116.4),
        Corner(60.1, -83.0, "Malmedy"),
        Straight(179.7),
        Corner(32.0, 91.1, "Bruxelles"),
        Straight(421.4, "descent to Pouhon"),
        Corner(149.8, 39.5, "Pouhon (1)"),
        Corner(119.9, 24.8, "Pouhon (2)"),
        Straight(205.9),
        Corner(50.0, -86.4, "Fagnes (1)"),
        Corner(60.0, 54.4, "Fagnes (2)"),
        Straight(300.1, "Curve Paul Frere"),
        Corner(85.0, -70.3, "Campus"),
        Straight(334.3),
        Corner(35.0, -129.4, "Stavelot"),
        Straight(1648.5, "Masta"),
        Corner(400.3, 19.8, "Blanchimont (1)"),
        Corner(500.7, 14.9, "Blanchimont (2)"),
        Straight(425.0, "to the Bus Stop"),
        Corner(25.0, -85.0, "Bus Stop (1)"),
        Corner(25.0, 85.0, "Bus Stop (2)"),
    ),
)

SILVERSTONE = Layout(
    name="silverstone",
    width=13.0,
    inspiration="Silverstone Grand Prix circuit",
    reference_length=5891.0,
    tags=("permanent", "high-speed", "medium"),
    segments=(
        Straight(396.2, "start/finish"),
        Corner(119.9, -59.3, "Abbey"),
        Straight(96.1),
        Corner(199.7, -44.6, "Farm Curve"),
        Straight(186.8),
        Corner(30.0, -97.7, "Village"),
        Corner(25.0, 130.0, "The Loop"),
        Straight(59.3),
        Corner(89.9, 55.7, "Aintree"),
        Straight(499.0, "Wellington Straight"),
        Corner(45.0, 86.1, "Brooklands"),
        Straight(64.5),
        Corner(40.0, -150.0, "Luffield"),
        Straight(96.4),
        Corner(299.6, -24.9, "Woodcote"),
        Straight(397.9, "to Copse"),
        Corner(130.0, -80.0, "Copse"),
        Straight(676.8, "to Maggotts"),
        Corner(150.0, 60.0, "Maggotts"),
        Corner(70.0, -70.0, "Becketts (1)"),
        Corner(60.0, 70.0, "Becketts (2)"),
        Corner(55.0, -65.0, "Becketts (3)"),
        Corner(90.1, -44.1, "Chapel"),
        Straight(933.8, "Hangar Straight"),
        Corner(100.2, -81.4, "Stowe"),
        Straight(626.3, "Vale"),
        Corner(45.0, 61.3, "Vale"),
        Corner(60.1, -106.1, "Club"),
        Straight(149.6),
    ),
)

CATALUNYA = Layout(
    name="catalunya",
    width=13.0,
    inspiration="Circuit de Barcelona-Catalunya",
    reference_length=4675.0,
    tags=("permanent", "technical", "medium"),
    segments=(
        Straight(693.5, "start/finish"),
        Corner(70.0, -69.7, "Elf"),
        Straight(124.5),
        Corner(55.0, -59.7, "Renault"),
        Straight(455.3),
        Corner(110.0, 71.1, "Repsol"),
        Straight(152.4),
        Corner(50.0, -68.7, "Seat"),
        Straight(237.8),
        Corner(200.3, -29.7, "Campsa"),
        Straight(142.2),
        Corner(90.1, 51.0, "La Caixa (entry)"),
        Corner(28.0, -95.0, "La Caixa"),
        Straight(873.2, "back straight"),
        Corner(120.3, -49.1, "Wurth"),
        Straight(142.9),
        Corner(35.0, -77.6, "Turn 10"),
        Straight(318.0),
        Corner(150.2, 55.9, "Turn 11"),
        Corner(60.0, -63.9, "Turn 12"),
        Straight(156.4),
        Corner(45.0, 90.0, "New Holland (1)"),
        Corner(45.0, -70.0, "New Holland (2)"),
        Straight(218.4),
        Corner(130.0, -44.7, "Turn 16"),
    ),
)

HOCKENHEIM = Layout(
    name="hockenheim",
    width=13.0,
    inspiration="Hockenheimring Grand Prix circuit",
    reference_length=4574.0,
    tags=("permanent", "high-speed", "hairpin", "medium"),
    segments=(
        Straight(567.3, "start/finish"),
        Corner(90.0, -55.0, "Nordkurve"),
        Straight(849.7, "Parabolika approach"),
        Corner(502.0, 30.6, "Parabolika (sweep)"),
        Corner(20.0, -144.9, "Spitzkehre"),
        Straight(527.3, "Mercedes-Tribune straight"),
        Corner(149.8, 62.0, "Mobil 1 Kurve (entry)"),
        Corner(45.0, -71.7, "Mobil 1 Kurve"),
        Straight(289.6),
        Corner(60.0, -53.1, "Turn 8"),
        Straight(238.0),
        Corner(120.0, 51.9, "Turn 9"),
        Corner(55.0, -61.8, "Turn 10"),
        Straight(318.1, "into the Motodrom"),
        Corner(70.1, -47.9, "Sudkurve (1)"),
        Corner(90.2, 80.4, "Sudkurve (2)"),
        Straight(163.7),
        Corner(40.0, -56.5, "Turn 13"),
        Straight(146.7),
        Corner(30.0, -65.0, "Turn 14"),
        Straight(286.0),
        Corner(160.4, -29.2, "Turn 15"),
    ),
)

ZANDVOORT = Layout(
    name="zandvoort",
    width=12.0,
    inspiration="Circuit Zandvoort",
    reference_length=4259.0,
    tags=("permanent", "narrow", "technical", "hard"),
    segments=(
        Straight(999.7, "start/finish"),
        Corner(28.0, -134.8, "Tarzanbocht"),
        Straight(166.4),
        Corner(60.2, 81.2, "Gerlachbocht"),
        Straight(221.9),
        Corner(22.0, -119.9, "Hugenholtzbocht"),
        Straight(150.9, "climb to Hunserug"),
        Corner(119.8, 62.6, "Hunserug"),
        Straight(206.4),
        Corner(69.9, -53.5, "Rob Slotemakerbocht"),
        Straight(154.8),
        Corner(139.5, -39.3, "Scheivlak"),
        Straight(145.8),
        Corner(79.7, 77.3, "Mastersbocht (1)"),
        Corner(49.9, -68.0, "Mastersbocht (2)"),
        Straight(145.3),
        Corner(59.9, -53.7, "Turn 10"),
        Straight(145.6),
        Corner(89.8, 67.4, "Turn 11"),
        Straight(126.6),
        Corner(44.9, -67.0, "Kumhobocht"),
        Straight(302.7),
        Corner(35.0, -52.4, "Turn 13"),
        Straight(198.9),
        Corner(65.0, -60.0, "Arie Luyendykbocht"),
        Straight(323.6),
    ),
)

HUNGARORING = Layout(
    name="hungaroring",
    width=12.0,
    inspiration="Hungaroring, Mogyorod",
    reference_length=4381.0,
    tags=("permanent", "technical", "low-speed", "hard"),
    segments=(
        Straight(992.3, "start/finish"),
        Corner(35.0, -150.2, "Turn 1"),
        Straight(233.1, "descent to Turn 2"),
        Corner(59.8, 61.2, "Turn 2"),
        Straight(82.1),
        Corner(89.7, -57.3, "Turn 3"),
        Straight(159.1),
        Corner(44.9, 66.7, "Turn 4"),
        Straight(82.7),
        Corner(69.8, -62.1, "Turn 5"),
        Corner(59.9, 48.4, "Turn 6"),
        Straight(88.3),
        Corner(79.7, -72.5, "Turn 7"),
        Straight(187.9),
        Corner(54.9, 58.9, "Turn 8"),
        Straight(85.0),
        Corner(40.0, -96.9, "Turn 9"),
        Straight(213.0),
        Corner(50.0, 65.2, "Turn 10"),
        Straight(141.7),
        Corner(25.0, -130.0, "Turn 11"),
        Straight(539.8),
        Corner(151.1, -39.9, "Turn 12"),
        Corner(60.2, 55.8, "Turn 13"),
        Straight(331.3),
        Corner(45.2, -107.4, "Turn 14"),
        Straight(233.6),
    ),
)

INTERLAGOS = Layout(
    name="interlagos",
    width=12.0,
    inspiration="Autodromo Jose Carlos Pace, Interlagos, anticlockwise",
    reference_length=4309.0,
    tags=("permanent", "technical", "anticlockwise", "hard"),
    segments=(
        Straight(137.9, "Reta Principal"),
        Corner(45.0, 65.0, "Curva do Senna (S do Senna 1)"),
        Corner(60.0, -65.0, "S do Senna 2"),
        Straight(68.8),
        Corner(196.8, 43.6, "Curva do Sol"),
        Straight(682.7, "Reta Oposta"),
        Corner(69.7, 48.1, "Descida do Lago (1)"),
        Corner(89.9, 34.0, "Descida do Lago (2)"),
        Straight(305.5),
        Corner(119.7, -42.3, "Ferradura (entry)"),
        Corner(64.9, 54.1, "Ferradura"),
        Straight(199.8),
        Corner(40.1, 40.9, "Laranja"),
        Straight(202.2),
        Corner(55.2, -75.0, "Pinheirinho (entry)"),
        Corner(30.0, 89.8, "Pinheirinho"),
        Straight(445.0),
        Corner(90.6, 35.9, "Bico de Pato (entry)"),
        Corner(28.0, 69.9, "Bico de Pato"),
        Straight(157.8),
        Corner(80.4, -62.1, "Mergulho"),
        Straight(557.7),
        Corner(60.0, 69.9, "Juncao"),
        Straight(304.3, "Subida dos Boxes"),
        Corner(298.3, 29.0, "Arquibancadas (1)"),
        Corner(394.6, 24.2, "Arquibancadas (2)"),
    ),
)

MONACO_STREET = Layout(
    name="monaco_street",
    width=9.0,
    inspiration=(
        "a narrow street circuit in the character of Monaco -- an original plan, "
        "not a reconstruction: Monaco's own layout runs closer to itself than a "
        "9 m wide annulus allows (see the module docstring)"
    ),
    reference_length=3255.0,
    tags=("street", "narrow", "hairpin", "low-speed", "hard"),
    segments=(
        Straight(162.4, "start/finish"),
        Corner(22.0, 107.9, "Turn 1"),
        Straight(415.8, "seafront"),
        Corner(400.0, 10.2, "Turn 2 (kink)"),
        Straight(362.6),
        Corner(45.0, 55.3, "Turn 3"),
        Straight(249.9),
        Corner(26.0, 53.4, "Turn 4"),
        Straight(182.8),
        Corner(30.0, -44.7, "Turn 5"),
        Straight(138.7),
        Corner(10.0, 126.5, "Hairpin"),
        Straight(39.8),
        Corner(28.0, 50.4, "Turn 7"),
        Straight(135.4),
        Corner(22.0, -99.5, "Turn 8"),
        Straight(109.1),
        Corner(24.0, 80.2, "Chicane (1)"),
        Straight(58.5),
        Corner(24.0, -89.9, "Chicane (2)"),
        Straight(145.7),
        Corner(35.0, 63.0, "Turn 11"),
        Straight(91.4),
        Corner(45.0, 92.3, "Pool complex (1)"),
        Straight(80.0),
        Corner(30.0, -90.0, "Pool complex (2)"),
        Straight(102.5),
        Corner(28.0, -82.2, "Pool complex (3)"),
        Straight(86.9),
        Corner(30.0, 85.3, "Pool complex (4)"),
        Straight(94.0),
        Corner(12.0, 52.2, "Turn 16"),
        Straight(181.0),
        Corner(26.0, -10.4, "Turn 17"),
    ),
)

LAYOUTS: dict[str, Layout] = {
    layout.name: layout
    for layout in (
        INDIANAPOLIS_OVAL,
        BRANDS_HATCH_INDY,
        RED_BULL_RING,
        MONZA,
        SPA_FRANCORCHAMPS,
        SILVERSTONE,
        CATALUNYA,
        HOCKENHEIM,
        ZANDVOORT,
        HUNGARORING,
        INTERLAGOS,
        MONACO_STREET,
    )
}

# Centerline samples per metre of track, for the spline `Track` builds. Fixed
# point counts are a trap across a library this varied: 2000 samples is 0.3 m
# on the 607 m sample oval and 3.5 m on Spa, and 3.5 m of granularity on the
# closest-centerline lookup is a third of a car width of error in the lateral
# offset the reward is computed from.
DEFAULT_ARC_STEP = 1.0


def get_layout(name: str) -> Layout:
    """Look up a layout by name.

    Args:
        name: Layout name, as used in `LAYOUTS`.

    Returns:
        The layout.

    Raises:
        KeyError: If no layout goes by that name.
    """
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(
            f"unknown track layout {name!r}; available: {', '.join(sorted(LAYOUTS))}"
        ) from None


def layouts_with_tags(*tags: str) -> list[Layout]:
    """Every layout carrying all of `tags`, in library order.

    Args:
        tags: Tags to require. With none, returns the whole library.

    Returns:
        Matching layouts.
    """
    required = set(tags)
    return [layout for layout in LAYOUTS.values() if required <= set(layout.tags)]


def build_track(name: str, arc_step: float = DEFAULT_ARC_STEP) -> Track:
    """Build a `Track` from a library layout.

    Args:
        name: Layout name.
        arc_step: Target spacing in metres between interpolated centerline
            samples.

    Returns:
        The track, ready to drive.
    """
    layout = get_layout(name)
    centerline, _ = build_centerline(layout)
    return Track(
        name=layout.name,
        centerline=centerline,
        width=layout.width,
        interpolation_resolution=resolution_for_arc_step(centerline, arc_step),
    )


def write_library(out_dir: str | Path) -> list[Path]:
    """Generate every library track as a JSON file.

    Args:
        out_dir: Directory to write into; created if missing.

    Returns:
        The written paths, in library order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in LAYOUTS:
        track = build_track(name)
        path = out_dir / f"{name}.json"
        save_track_json(track, str(path))
        written.append(path)
    return written


def main() -> None:
    """CLI: regenerate the track JSON files from the layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="data/tracks",
        help="Directory to write the track JSON files into (default: data/tracks)",
    )
    args = parser.parse_args()

    for path in write_library(args.out):
        layout = get_layout(path.stem)
        track = build_track(layout.name)
        print(
            f"{layout.name:22s} {track.lap_length:7.0f} m  "
            f"{layout.num_corners:2d} corners  "
            f"min radius {layout.min_radius:5.0f} m  "
            f"width {layout.width:4.1f} m  -> {path}"
        )


if __name__ == "__main__":
    main()
