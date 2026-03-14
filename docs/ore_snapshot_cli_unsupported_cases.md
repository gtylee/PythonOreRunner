# ORE Snapshot CLI Unsupported Cases

Final status:
- no example cases remain in a separate unsupported-product/model failure bucket
- unsupported Python-path products now fall back to ORE reference mode when the
  case ships usable `Output` or `ExpectedOutput` data

Current scan result:
- hard failures from unsupported product/model assumptions: `0`

Notes:
- this does not mean all products are natively supported by the Python snapshot path
- it means the shipped example set no longer has any cases that fail solely for
  that reason

If future examples are added without usable native/reference outputs, recreate
this file as a live exclusion list.
