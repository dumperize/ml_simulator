"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from user_input.metrics import Metric

import pandas as pd

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"
    memo: Dict = field(default_factory=dict)

    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {}

        key = "+".join(tables.keys())
        if self.memo.get(key):
            return self.memo[key]

        report = self.report_

        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")

        result = []
        passed = 0
        failed = 0
        errors = 0
        for table_name, checker_class, limits in self.checklist:
            df = tables[table_name]
            error = ''
            try:
                values = checker_class(df)
                if len(limits) == 0: 
                    status = '.'
                    passed += 1
                else:
                    for limit in limits:
                        if values[limit] >= limits[limit][0] and values[limit] <= limits[limit][1]:
                            status = '.'
                            passed += 1
                        else:
                            status = 'F'
                            failed += 1
            except Exception as e:
                status = 'E'
                error = type(e).__name__
                errors += 1
            result.append({
                "table_name": table_name,
                "metric": str(checker_class),
                "limits": str(limits),
                "values": values,
                "status": status,
                "error": error
            })
        report['title'] = 'DQ Report for tables ' + sorted(list(tables.keys())).__str__()
        report['result'] = pd.DataFrame(result)

        report['total'] = len(result)

        report['passed'] = passed
        report['passed_pct'] = round(passed / report['total'] * 100, 2)
        report['failed'] = failed
        report['failed_pct'] = round(failed / report['total'] * 100, 2)
        report['errors'] = errors
        report['errors_pct'] = round(errors / report['total'] * 100, 2)

        self.memo[key] = report
        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
