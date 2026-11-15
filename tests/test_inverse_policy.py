"""Fixed bisection metadata must not inherit path-sum replay tolerance."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from warplab.inverse import solve_inverse
from warplab.inverse_oracle import verify_inverse


class InversePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controls = json.loads((Path(__file__).parents[1] / 'docs/inverse-controls.json').read_text())
        cls.report = solve_inverse(controls[0]['input'])
        assert cls.report['trace']

    def test_fixed_policy_constants_reject_even_one_ulp_changes(self):
        for key in ('absolute_mean_tolerance', 'penalty_width_tolerance'):
            value = self.report[key]
            for changed in (math.nextafter(value, 0), math.nextafter(value, math.inf),
                            1e-7, 0, True, None, '1e-9', math.inf):
                report = copy.deepcopy(self.report)
                report[key] = changed
                with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                    verify_inverse(report)

    def test_every_dyadic_trace_coordinate_and_selected_penalty_is_exact(self):
        paths = [('trace', i, key) for i in range(len(self.report['trace']))
                 for key in ('lower', 'upper', 'penalty')]
        paths.append(('selected', 'penalty'))
        for path in paths:
            for toward in (-math.inf, math.inf):
                report = copy.deepcopy(self.report)
                at = report
                for key in path[:-1]:
                    at = at[key]
                at[path[-1]] = math.nextafter(at[path[-1]], toward)
                with self.subTest(path=path, toward=toward), self.assertRaises(ValueError):
                    verify_inverse(report)

    def test_computed_path_sum_fields_keep_numerical_replay_tolerance(self):
        report = copy.deepcopy(self.report)
        report['endpoints'][0]['mean'] += 1e-8
        self.assertTrue(verify_inverse(report))

    def test_cli_rejects_policy_and_geometry_forgeries(self):
        for path in [('absolute_mean_tolerance',), ('trace', 0, 'penalty')]:
            report = copy.deepcopy(self.report)
            at = report
            for key in path[:-1]:
                at = at[key]
            at[path[-1]] += 1e-8
            result = subprocess.run([sys.executable, '-m', 'warplab', 'verify-inverse', '-'],
                                    input=json.dumps(report), text=True, capture_output=True)
            with self.subTest(path=path):
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
