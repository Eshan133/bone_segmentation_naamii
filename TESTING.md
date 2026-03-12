# SamanKinMel Testing Guide

## Purpose

This document defines how to test the SamanKinMel web app consistently and document results in a professional, repeatable way.

The application is a static front-end site with browser-side state only. There is no backend, database, or API integration in this repository. Application data and session state are stored in browser `localStorage`.

## Application Summary

- Application type: Static HTML/CSS/JavaScript marketplace demo
- Primary entry point: `index.html`
- Runtime dependency: Browser + local HTTP server
- Data persistence: `localStorage`
- Roles supported:
  - Buyer
  - Seller
  - Admin

## Scope

### In Scope

- Public storefront flows
- Product discovery
- Product detail behavior
- Cart and checkout flow
- Registration and login
- Seller dashboard flows
- Admin panel access and basic actions
- Browser persistence and navigation
- Responsive smoke testing

### Out of Scope

- Real payment gateway integration
- Real order management backend
- Real product moderation backend
- Real analytics
- Real email, OTP, password reset, or notifications service
- Security validation beyond front-end behavior

## Test Environment

### Recommended Browsers

- Google Chrome latest stable
- Microsoft Edge latest stable
- Safari latest stable on macOS
- Firefox latest stable

### Recommended Viewports

- Desktop: `1440 x 900`
- Tablet: `768 x 1024`
- Mobile: `390 x 844`

### Local Run Command

Use a local HTTP server instead of opening files directly:

```bash
cd /Users/ishanmaharjan/Downloads/samankinmel
npx live-server . --port=8000
```

Alternative:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## Test Data Management

The app persists state in browser `localStorage`. Testing must start from a known state.

### Keys Used by the App

- `cart`
- `appliedDiscount`
- `users`
- `currentUser`
- `orders`
- `lastOrder`
- `customProducts`
- `darkMode`

### Reset Procedure

Before each clean test cycle:

1. Open browser DevTools.
2. Go to Application/Storage.
3. Clear site `localStorage`.
4. Refresh the app.

Browser console alternative:

```js
localStorage.clear();
location.reload();
```

## Demo Accounts

These are created automatically when using the demo login buttons on `login.html`.

| Role | Email | Password |
| --- | --- | --- |
| Buyer | `buyer@demo.com` | `demo123` |
| Seller | `seller@demo.com` | `demo123` |
| Admin | `admin@samankinmel.com` | `demo123` |

## Test Approach

### 1. Smoke Testing

Goal: confirm the app loads and core navigation works.

Minimum smoke path:

1. Open home page.
2. Browse products.
3. Open a product detail page.
4. Add item to cart.
5. Complete checkout.
6. Confirm order confirmation page renders.
7. Log in as seller and open dashboard.
8. Log in as admin and open admin panel.

### 2. Functional Testing

Goal: verify each user-facing workflow against expected behavior.

### 3. Regression Testing

Goal: rerun critical cases after any UI or JavaScript change, especially around:

- authentication
- cart totals
- checkout
- `localStorage` state
- seller/admin access control

### 4. Exploratory Testing

Goal: intentionally try invalid, unexpected, or sequence-based actions.

Examples:

- checkout with empty cart
- repeat coupon application
- refresh mid-checkout
- role switching within the same browser session
- direct URL access to protected pages

## Functional Test Matrix

| Area | Priority | Notes |
| --- | --- | --- |
| Home page rendering | High | Core entry point |
| Product listing and filters | High | Main discovery flow |
| Product detail and quantity | High | Add-to-cart path |
| Cart calculations | High | Business-critical |
| Checkout and order placement | High | Business-critical |
| Registration and login | High | Role access depends on it |
| Seller dashboard | Medium | Mostly demo behavior |
| Admin panel | Medium | Mostly demo behavior |
| Theme toggle and greeting | Low | UX feature only |
| Responsive layout | Medium | Required for presentation quality |

## Detailed Test Cases

### TC-01 Home Page Load

- Objective: Verify the landing page renders without visible breakage.
- Preconditions: Clean `localStorage`.
- Steps:
  1. Open `/index.html`.
  2. Verify navbar, hero section, category cards, featured products, testimonials, and footer render.
  3. Verify featured products are populated.
- Expected Result:
  - No broken layout
  - Featured products visible
  - Cart badge shows `0`

### TC-02 Home Search Redirect

- Objective: Verify home search sends the user to product listing with query parameter.
- Steps:
  1. Enter a valid product keyword in the home search bar.
  2. Click search.
- Expected Result:
  - Browser navigates to `products.html?search=...`
  - Products are filtered to matching items

### TC-03 Category Shortcut

- Objective: Verify category cards apply category filters.
- Steps:
  1. Click each category card from the home page.
- Expected Result:
  - Browser navigates to `products.html?category=...`
  - Matching checkbox is selected
  - Product grid is filtered

### TC-04 Product Listing Filters

- Objective: Verify search, category, rating, location, price, and sort controls.
- Steps:
  1. Open `/products.html`.
  2. Apply one filter at a time.
  3. Combine multiple filters.
  4. Use Reset Filters.
- Expected Result:
  - Product count updates
  - Product list changes accordingly
  - Reset restores full list

### TC-05 Product Detail Page

- Objective: Verify product detail rendering and quantity control.
- Steps:
  1. Open any product card.
  2. Verify title, seller, location, stock, category, and reviews section.
  3. Increase and decrease quantity.
  4. Attempt to go below `1`.
- Expected Result:
  - Product detail page loads
  - Quantity never drops below `1`
  - Related products render

### TC-06 Add to Cart From Listing

- Objective: Verify cart addition from product cards.
- Steps:
  1. Add the same product twice from listing view.
- Expected Result:
  - Cart badge increments
  - Cart stores one line item with increased quantity

### TC-07 Add to Cart From Product Detail

- Objective: Verify quantity-based add to cart.
- Steps:
  1. Set quantity greater than `1`.
  2. Click Add to Cart.
- Expected Result:
  - Cart quantity reflects selected amount

### TC-08 Cart Management

- Objective: Verify cart quantities, removal, subtotal, delivery, and total.
- Steps:
  1. Open `/cart.html`.
  2. Increase quantity.
  3. Decrease quantity.
  4. Remove an item.
  5. Validate totals after each step.
- Expected Result:
  - Subtotal updates correctly
  - Delivery is `FREE` above NPR 5000
  - Delivery is NPR 150 at or below NPR 5000
  - Empty-cart state appears when all items are removed

### TC-09 Coupon Application

- Objective: Verify supported coupon codes and invalid-code handling.
- Valid Codes:
  - `NEPAL10`
  - `FIRSTORDER`
  - `NAMASTE20`
  - `SAMANKINMEL`
- Steps:
  1. Add products to cart.
  2. Apply each valid coupon in a clean run.
  3. Apply an invalid coupon.
- Expected Result:
  - Valid codes reduce total
  - Discount row becomes visible
  - Invalid code shows an error notification

### TC-10 Buyer Registration

- Objective: Verify buyer registration flow.
- Steps:
  1. Open `/register.html`.
  2. Leave role as buyer.
  3. Complete all required fields.
  4. Submit.
- Expected Result:
  - Account is created
  - User is stored in `localStorage.users`
  - `currentUser` is set
  - User redirects to home page

### TC-11 Seller Registration

- Objective: Verify seller registration flow.
- Steps:
  1. Open `/register.html`.
  2. Switch to seller role.
  3. Complete required fields plus seller details.
  4. Submit.
- Expected Result:
  - Seller account is created
  - User redirects to seller dashboard

### TC-12 Duplicate Registration and Password Mismatch

- Objective: Verify front-end validation for duplicate email and mismatched password.
- Steps:
  1. Attempt registration with mismatched passwords.
  2. Register a user.
  3. Attempt registration again with the same email.
- Expected Result:
  - Mismatch is blocked
  - Duplicate email is blocked

### TC-13 Demo Login

- Objective: Verify demo login buttons for all roles.
- Steps:
  1. Open `/login.html`.
  2. Use Buyer demo login.
  3. Log out.
  4. Use Seller demo login.
  5. Log out.
  6. Use Admin demo login.
- Expected Result:
  - Buyer lands on home page
  - Seller lands on dashboard
  - Admin lands on admin panel

### TC-14 Manual Login

- Objective: Verify login with a registered user.
- Steps:
  1. Register a user.
  2. Log out.
  3. Log in with the same credentials.
- Expected Result:
  - Login succeeds
  - Role-based redirect works

### TC-15 Route Protection

- Objective: Verify protected pages redirect unauthenticated users.
- Steps:
  1. Clear `currentUser`.
  2. Open `/dashboard.html`.
  3. Open `/admin.html`.
- Expected Result:
  - Dashboard redirects to login unless logged in as seller/admin
  - Admin redirects to login unless logged in as admin

### TC-16 Checkout Flow

- Objective: Verify end-to-end checkout.
- Steps:
  1. Add items to cart.
  2. Open checkout.
  3. Complete shipping form.
  4. Select each payment method across separate runs.
  5. Place order.
- Expected Result:
  - Order is stored in `localStorage.orders`
  - `lastOrder` is stored
  - `cart` and `appliedDiscount` are removed
  - User is redirected to `/order-confirm.html`

### TC-17 Checkout Prefill

- Objective: Verify checkout uses logged-in user data when available.
- Steps:
  1. Log in or register.
  2. Add item to cart and open checkout.
- Expected Result:
  - First name, last name, email, and phone are prefilled

### TC-18 Seller Dashboard Access

- Objective: Verify seller dashboard loads for seller and admin roles.
- Steps:
  1. Log in as seller.
  2. Open `/dashboard.html`.
  3. Repeat as admin.
- Expected Result:
  - Seller can access dashboard
  - Admin can also access dashboard in current implementation

### TC-19 Seller Add Product

- Objective: Verify seller can create a custom product entry.
- Steps:
  1. Log in as seller.
  2. Open Add Product tab.
  3. Submit a new product.
  4. Open My Products tab.
- Expected Result:
  - Product appears in seller product list
  - Product persists after refresh

### TC-20 Admin Panel Access and Basic Actions

- Objective: Verify admin panel access and primary tabs.
- Steps:
  1. Log in as admin.
  2. Open each tab.
  3. Click Approve, Remove, and Suspend actions.
- Expected Result:
  - Admin panel loads
  - Tab switching works
  - Action buttons show notifications

### TC-21 Dark Mode Persistence

- Objective: Verify theme preference persists.
- Steps:
  1. Toggle dark mode.
  2. Refresh.
- Expected Result:
  - Theme persists using `localStorage.darkMode`

### TC-22 Responsive Smoke Test

- Objective: Verify the app remains usable on smaller screens.
- Steps:
  1. Test home, products, cart, login, and checkout on mobile viewport.
  2. Verify navigation and forms remain usable.
- Expected Result:
  - No blocking overlap or clipped actions
  - Primary flows remain accessible

## Expected Storage Behavior

Use this section while validating browser state.

| Key | Created By | Expected Behavior |
| --- | --- | --- |
| `cart` | Add to cart | Stores line items with quantity |
| `appliedDiscount` | Coupon apply | Stores discount amount, not coupon code |
| `users` | Register or demo login | Stores locally created/demo users |
| `currentUser` | Login or register | Controls role-based access |
| `orders` | Place order | Accumulates submitted orders |
| `lastOrder` | Place order | Drives order confirmation page |
| `customProducts` | Seller add product | Stores seller-created products |
| `darkMode` | Theme toggle | Persists theme preference |

## Known Limitations and Current Risks

These are implementation realities to account for during testing, not assumptions.

1. Seller-added products are stored in `customProducts`, but public product listing uses only the hardcoded `PRODUCTS` array, so new seller products do not appear on the storefront. See [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L397), [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1148), [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1217).
2. Seller-added image URLs are not saved because the add-product handler does not include `prodImage` in the new product object. See [dashboard.html](/Users/ishanmaharjan/Downloads/samankinmel/dashboard.html#L157) and [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1201).
3. Seller dashboard statistics are hardcoded and do not reflect actual `orders` or `customProducts`. See [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1128).
4. Admin overview, user table, and order table are mostly static markup and are not driven by live `localStorage` data. See [admin.html](/Users/ishanmaharjan/Downloads/samankinmel/admin.html#L47), [admin.html](/Users/ishanmaharjan/Downloads/samankinmel/admin.html#L116), [admin.html](/Users/ishanmaharjan/Downloads/samankinmel/admin.html#L166).
5. Checkout can proceed even if the cart is empty; there is no front-end guard in `placeOrder()`. See [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L929).
6. Seller and admin tab actions mostly show notifications and do not persist changes. See [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1228), [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1259), [js/app.js](/Users/ishanmaharjan/Downloads/samankinmel/js/app.js#L1298).

## Suggested Regression Suite

Run these after every functional change:

1. Home page load
2. Browse products and filters
3. Product detail to cart
4. Cart quantity and totals
5. Coupon application
6. Checkout and order confirmation
7. Buyer register and login
8. Seller login and add product
9. Admin login and tab switching
10. Dark mode persistence

## Evidence Collection

For each executed test case, record:

- Test case ID
- Browser and version
- Device or viewport
- Preconditions
- Actual result
- Pass/fail status
- Screenshot or screen recording path
- Defect ID if failed

Use [TEST-REPORT.md](/Users/ishanmaharjan/Downloads/samankinmel/docs/TEST-REPORT.md) for execution tracking.
