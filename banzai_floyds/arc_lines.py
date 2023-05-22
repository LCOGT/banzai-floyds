"""
Line lists from FLOYDS arc frames:
Wavelengths, source, notes, and used lines are sourced from FLOYDS_lines.text
Strengths are determined for the 2" slit using ogg2m001-en06-20220222-0036-a00.fits and are relative to strongest line.
Blue lines have been calibrated to match the red strengths based on the overlapping 5460.75 line.
Lines without strengths could not be found, possibly too weak or blended with nearby lines.
"""
from astropy.table import Table

used_lines = [{'wavelength': 3650.158, 'line_strength': 0.0556, 'line_source': 'Hg', 'line_notes': ''},
              {'wavelength': 4046.565, 'line_strength': 0.0659, 'line_source': 'Hg', 'line_notes': ''},
              {'wavelength': 4077.837, 'line_strength': 0.0056, 'line_source': 'Hg', 'line_notes': ''},
              {'wavelength': 4358.335, 'line_strength': 0.2147, 'line_source': 'Hg', 'line_notes': ''},
              {'wavelength': 5460.75, 'line_strength': 0.1784, 'line_source': 'Hg',
               'line_notes': 'Found in both Red/Blue Orders'},
              {'wavelength': 5769.61, 'line_strength': 0.0296, 'line_source': 'Hg', 'line_notes': ''},
              {'wavelength': 6677.282, 'line_strength': 0.0017, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 6965.4307, 'line_strength': 0.2379, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 7067.2175, 'line_strength': 0.0694, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 7147.0416, 'line_strength': 0.0113, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 7272.9359, 'line_strength': 0.0599, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 7635.1056, 'line_strength': 0.988, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 7948.1764, 'line_strength': 0.1968, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 8264.5225, 'line_strength': 0.3103, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 8521.4422, 'line_strength': 0.1409, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 8667.9442, 'line_strength': 0.0321, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 9122.9674, 'line_strength': 1.0, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 9224.4992, 'line_strength': 0.1173, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 9657.7863, 'line_strength': 0.151, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 9784.5028, 'line_strength': 0.0063, 'line_source': 'ArI', 'line_notes': ''},
              {'wavelength': 10139.75, 'line_strength': 0.005, 'line_source': 'Hg', 'line_notes': ''},
              ]

unused_lines = [{'wavelength': 3451.689, 'line_strength': 'nan', 'line_source': 'Hg',
                 'line_notes': 'No flux in FLOYDS'},
                {'wavelength': 3654.842, 'line_strength': 'nan', 'line_source': 'Hg', 'line_notes': ''},
                {'wavelength': 3663.2793, 'line_strength': 0.0064, 'line_source': 'Hg', 'line_notes': 'blend'},
                {'wavelength': 4339.2232, 'line_strength': 'nan', 'line_source': 'Hg', 'line_notes': 'Weak line'},
                {'wavelength': 4347.5, 'line_strength': 'nan', 'line_source': 'Hg', 'line_notes': 'Weak line'},
                {'wavelength': 4916.068, 'line_strength': 'nan', 'line_source': 'Hg', 'line_notes': 'Weak line'},
                {'wavelength': 5790.67, 'line_strength': 'nan', 'line_source': 'Hg', 'line_notes': 'Blend?'},
                {'wavelength': 6416.307, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 5675.81, 'line_strength': 'nan', 'line_source': 'Hg', 'line_notes': 'Weak'},
                {'wavelength': 4680.1359, 'line_strength': 'nan', 'line_source': 'Zn',
                 'line_notes': 'Removed until we start turning on the Zn lamp again.'},
                {'wavelength': 4722.1569, 'line_strength': 'nan', 'line_source': 'Zn',
                 'line_notes': 'Removed until we start turning on the Zn lamp again.'},
                {'wavelength': 4810.5321, 'line_strength': 'nan', 'line_source': 'Zn',
                 'line_notes': 'Removed until we start turning on the Zn lamp again.'},
                {'wavelength': 5181.9819, 'line_strength': 'nan', 'line_source': 'Zn',
                 'line_notes': 'Removed until we start turning on the Zn lamp again.'},
                {'wavelength': 6114.9232, 'line_strength': 'nan', 'line_source': 'ArII', 'line_notes': ''},
                {'wavelength': 6752.8335, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 6871.2891, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 6766.6117, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 6827.2529, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 6937.6642, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 7206.9804, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 7372.1184, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 7383.9805, 'line_strength': 0.1489, 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 7503.8691, 'line_strength': 0.253, 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 7514.6518, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 7723.7599, 'line_strength': 0.3542, 'line_source': 'ArI', 'line_notes': 'Blend'},
                {'wavelength': 8006.1567, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 8014.7857, 'line_strength': 0.2952, 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 8103.6931, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 8115.3108, 'line_strength': 0.5181, 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 8408.2096, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 8424.6475, 'line_strength': 0.4551, 'line_source': 'ArI',
                 'line_notes': 'Blended in FLOYDS'},
                {'wavelength': 9194.6385, 'line_strength': 0.0023, 'line_source': 'ArI',
                 'line_notes': 'Too weak in FLOYDS'},
                {'wavelength': 9291.5313, 'line_strength': 0.0017, 'line_source': 'ArI', 'line_notes': 'Weak'},
                {'wavelength': 9354.2198, 'line_strength': 0.0077, 'line_source': 'ArI', 'line_notes': 'Weak'},
                {'wavelength': 10052.1, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': 'Weak'},
                {'wavelength': 10332.76, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': 'Weak'},
                {'wavelength': 10470.054, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 10506.47, 'line_strength': 'nan', 'line_source': 'ArI', 'line_notes': ''},
                {'wavelength': 10673.55, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'Too weak to get a reliable centroid'},
                {'wavelength': 10950.74, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'No flux in FLOYDS'},
                {'wavelength': 11078.87, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'No flux in FLOYDS'},
                {'wavelength': 11106.44, 'line_strength': 'nan', 'line_source': 'ArI',
                 'line_notes': 'No flux in FLOYDS'}
                ]


def arc_lines_table():
    wavelength = []
    strength = []
    source = []
    notes = []
    used = []
    for line in used_lines:
        wavelength.append(line['wavelength'])
        strength.append(float(line['line_strength']))
        source.append(line['line_source'])
        notes.append(line['line_notes'])
        used.append(True)

    for line in unused_lines:
        wavelength.append(line['wavelength'])
        strength.append(float(line['line_strength']))
        source.append(line['line_source'])
        notes.append(line['line_notes'])
        used.append(False)

    lines_table = Table({'wavelength': wavelength,
                         'strength': strength,
                         'source': source,
                         'notes': notes,
                         'used': used})

    return lines_table
