# SamanKinMel Test Report Template

## Test Run Summary

- Project: SamanKinMel
- Build/Version: Workspace snapshot on March 12, 2026
- Test Date: March 12, 2026
- Tester: Codex
- Environment: Local execution against static site served over HTTP
- Browser(s): Playwright Chromium 145.0.7632.6
- Viewport(s): 1440 x 900
- Server Command: `python3 -m http.server 8000`

## Scope Covered

- Smoke testing
- Functional testing
- Regression testing
- Exploratory testing

## Overall Status

- Total test cases executed: 23
- Passed: 22
- Failed: 1
- Blocked: 0
- Not run: 0

## Key Findings

1. Full test matrix is now executed. Core storefront, authentication, seller/admin access, theme persistence, and mobile smoke behavior all passed except for one checkout defect.
2. Checkout allows an order to be placed with an empty cart, which is a release-blocking functional defect for transaction flow.
3. Seller-created products persist locally and appear in the seller dashboard, but they do not appear on the public storefront, and the product image URL field is not saved.

## Defect Summary

| Defect ID | Title | Severity | Priority | Status | Test Case |
| --- | --- | --- | --- | --- | --- |
| DEF-001 | Seller-created products do not appear on public storefront | Medium | Medium | Open | TC-19 |
| DEF-002 | Seller product image URL is not saved | Low | Medium | Open | TC-19 |
| DEF-003 | Checkout allows order placement with an empty cart | High | High | Open | EXP-01 |

## Execution Log

| Test Case | Area | Preconditions | Actual Result | Status | Evidence | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TC-01 | Home Page Load | Clean localStorage | Home page rendered, featured products loaded, cart badge started at 0. | Pass | `test-artifacts/TC-01-home.png` |  |
| TC-02 | Home Search Redirect | Home page loaded | Search redirected to products page and filtered results rendered. Observed count: 15 products found. | Pass | `test-artifacts/first-pass-results.json` | Search term used: `tea` |
| TC-03 | Category Shortcut | Home page loaded | Category card redirected to products page and auto-selected matching filter. | Pass | `test-artifacts/first-pass-results.json` | Verified `handcraft` route/filter sync |
| TC-04 | Product Listing Filters | Products page loaded | Food filter changed product count from 15 products found to 4 products found, reset restored 15 products found. | Pass | `test-artifacts/first-pass-results.json` |  |
| TC-05 | Product Detail Page | Product list available | Product detail rendered with quantity guard and related products section. | Pass | `test-artifacts/first-pass-results.json` | Quantity stayed within lower bound of 1 |
| TC-06 | Add to Cart From Listing | Product list available | Repeated add-to-cart increments quantity on a single line item. | Pass | `test-artifacts/first-pass-results.json` | Cart badge reached 2 |
| TC-07 | Add to Cart From Product Detail | Product detail page loaded | Product detail add-to-cart respected selected quantity of 3. | Pass | `test-artifacts/second-pass-results.json` |  |
| TC-08 | Cart Management | Cart contains items | Cart totals updated from NPR 4,700 to NPR 1,200, item removal worked. | Pass | `test-artifacts/first-pass-results.json` |  |
| TC-09 | Coupon Application | Cart contains items | Valid coupon applied and stored `appliedDiscount=350`. | Pass | `test-artifacts/first-pass-results.json` | Coupon used: `NEPAL10` |
| TC-10 | Buyer Registration | Clean localStorage | Buyer registration stored user and redirected to home page. | Pass | `test-artifacts/second-pass-results.json` | Test account: `buyer.qa@example.com` |
| TC-11 | Seller Registration | Clean localStorage | Seller registration stored user and redirected to seller dashboard. | Pass | `test-artifacts/second-pass-results.json` | Test account: `seller.qa@example.com` |
| TC-12 | Duplicate Registration and Password Mismatch | Registration page available | Registration blocked mismatched passwords and duplicate email reuse. | Pass | `test-artifacts/second-pass-results.json` |  |
| TC-13 | Demo Login | Login page loaded | Buyer, seller, and admin demo logins redirected to their expected destinations. | Pass | `test-artifacts/first-pass-results.json` |  |
| TC-14 | Manual Login | Registered user exists | Registered buyer logged in successfully with manual credentials. | Pass | `test-artifacts/second-pass-results.json` | Test account: `manual.qa@example.com` |
| TC-15 | Route Protection | currentUser cleared | Unauthenticated access to seller and admin routes redirected to login. | Pass | `test-artifacts/first-pass-results.json` |  |
| TC-16 | Checkout Flow | Cart contains items | Checkout stored 1 order, cleared cart, and redirected to confirmation. | Pass | `test-artifacts/first-pass-results.json` | Payment path exercised through default selected option |
| TC-17 | Checkout Prefill | Logged-in user exists | Checkout prefilled buyer first name, email, and phone from `currentUser`. | Pass | `test-artifacts/first-pass-results.json` |  |
| TC-18 | Seller Dashboard Access | Seller/admin user exists | Seller and admin both accessed dashboard per current route-guard behavior. | Pass | `test-artifacts/second-pass-results.json` | Current implementation permits admin access to seller dashboard |
| TC-19 | Seller Add Product | Seller logged in | Seller-created product persisted to `customProducts` and appeared in seller product grid. | Pass | `test-artifacts/first-pass-results.json` | Follow-on defects DEF-001 and DEF-002 logged |
| TC-20 | Admin Panel Access and Basic Actions | Admin logged in | Admin access, tab switching, and action notification were verified. | Pass | `test-artifacts/first-pass-results.json` |  |
| TC-21 | Dark Mode Persistence | Home or products page loaded | Dark mode preference persisted and reapplied on reload. | Pass | `test-artifacts/second-pass-results.json` |  |
| TC-22 | Responsive Smoke Test | Browser device toolbar enabled | Mobile viewport preserved navigation access and checkout primary action visibility. | Pass | `test-artifacts/TC-22-mobile.png` | Viewport: 390 x 844 |
| EXP-01 | Exploratory: Empty Cart Checkout | Clean localStorage, direct access to checkout | Order confirmation was reachable with no cart items. | Fail | `test-artifacts/first-pass-results.json` | Logged as DEF-003 |

## Open Risks

1. Admin and seller management actions were only verified at notification level; persistence and data integrity remain limited by current demo implementation.
2. Responsive coverage in this run was smoke-level at one mobile viewport, not a full layout audit across multiple devices and pages.
3. The app remains fully front-end and `localStorage`-driven, so no server-side validation, payment, or authorization guarantees were assessed.

## Recommendation

- Release recommendation: Do not treat this build as production-ready.
- Conditions or follow-up required:
  - Fix DEF-003 before any demonstration that includes checkout credibility.
  - Decide whether seller-created products are expected to appear publicly; if yes, fix DEF-001.
  - Save seller image URLs or remove the form field until implemented.
  - After defect fixes, rerun checkout, seller product visibility, and seller product creation regression cases.
