"""核对易出错的数据同步和CSV契约，不训练模型。"""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline'))
from depth_ir_io import read_modalities
from analyze_predictions import analyze


class Contracts(unittest.TestCase):
    def test_short_clip_repeats_both_modalities_together(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'Depth_Color').mkdir()
            (root / 'IR').mkdir()
            Image.new('RGB', (10, 8), (255, 0, 0)).save(root / 'Depth_Color/Depth_00000062_Color.png')
            Image.new('L', (10, 8), 127).save(root / 'IR/IR_00000062.png')
            x = read_modalities(root, frames=16, size=8)
            self.assertEqual(x.shape, (4, 16, 8, 8))
            np.testing.assert_allclose(x[:, 0], x[:, -1])
            self.assertAlmostEqual(float(x[3, 0, 0, 0]), 127 / 255, places=6)

    def test_same_length_different_frame_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['Depth_Color', 'IR']:
                (root / name).mkdir()
            Image.new('RGB', (8, 8)).save(root / 'Depth_Color/Depth_00000062_Color.png')
            Image.new('L', (8, 8)).save(root / 'IR/IR_00000063.png')
            with self.assertRaisesRegex(ValueError, 'identical ordered'):
                read_modalities(root)

    def test_anonymous_frame_names_used_by_judges(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, mode in [('Depth_Color', 'RGB'), ('IR', 'L')]:
                (root / name).mkdir()
                Image.new(mode, (8, 8)).save(root / name / '000000.png')
            self.assertEqual(read_modalities(root, frames=2, size=8).shape, (4, 2, 8, 8))

    def test_csv_id_contract_and_fixed_40_class_macro_f1(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'modality': 'depth_ir', 'classes': [str(i) for i in range(40)],
                                            'samples': [{'file': 'abc', 'label': 0}, {'file': 'def', 'label': 1}]}), encoding='utf-8')
            predictions = root / 'predictions.csv'

            def emit(rows):
                with predictions.open('w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['sample_id', 'predicted_class'])
                    writer.writerows(rows)

            emit([['abc.npy', 0], ['def.npy', 1]])
            result = analyze(predictions, manifest)
            self.assertEqual(result['metrics']['accuracy'], 100)
            self.assertEqual(result['metrics']['macro_f1'], 5)  # 2/40，不是仅对已出现类别平均。
            for rows in [[['abc', 0], ['def', 1]], [['abc.npy', 0]],
                         [['abc.npy', 0], ['abc.npy', 1]], [['abc.npy', 40], ['def.npy', 1]]]:
                emit(rows)
                with self.assertRaises(ValueError):
                    analyze(predictions, manifest)


if __name__ == '__main__':
    unittest.main()
