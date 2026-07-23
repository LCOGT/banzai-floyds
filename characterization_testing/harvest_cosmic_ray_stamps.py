"""harvest real cosmic-ray morphology stamps and save them for unit testing.

Take Each stamp is a JSON object with `flux` (still int from raw) and `flagged`
 (Pixels that have >5-sigma difference between the pair of frames are set to 1)
All the pairs used here are public data (more than a year old).
"""
import json
import os

import numpy as np

from astroscrappy_study import download_frame, load_frame, CR_FRAME_PAIRS, harvest_cr_stamps

OUTPUT_PATH = os.path.join('..', 'banzai_floyds', 'tests', 'data', 'cosmic_ray_stamps.json')

if __name__ == '__main__':
    stamps = []
    for basename1, basename2 in CR_FRAME_PAIRS:
        frame1 = load_frame(download_frame(basename1))
        frame2 = load_frame(download_frame(basename2))
        stamps += harvest_cr_stamps(frame1, frame2)
    print(f'Harvested {len(stamps)} real cosmic-ray stamps from the quiet, off-order region')

    stamps_json = [{'flux': np.round(stamp['flux']).astype(int).tolist(),
                    'flagged': stamp['flagged'].astype(int).tolist()} for stamp in stamps]
    with open(OUTPUT_PATH, 'w') as output_file:
        json.dump(stamps_json, output_file)
    print(f'Wrote {len(stamps)} stamps to {OUTPUT_PATH}')
