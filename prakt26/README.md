# 🎮 CyberSportsPortal API Tests

![Bruno](https://img.shields.io/badge/Bruno-API%20Client-orange)
![OpenAPI](https://img.shields.io/badge/OpenAPI-spec-6BA539?logo=openapiinitiative&logoColor=white)
![Requests](https://img.shields.io/badge/requests-11%20GET-blue)

[🇷🇺 Русский](README.md) · 🇬🇧 English

**API test collection for an esports portal**, built in [Bruno](https://www.usebruno.com/)
(VS Code extension or desktop app) against an OpenAPI specification.
Internship at **Reksoft** — "Databases" module.

## 📋 Overview

The collection covers **every GET endpoint of the test server — 11 requests**. Each request includes:

- ✅ at least **2 assertions** — status code and response structure checks;
- 🧪 **JavaScript tests** validating response fields;
- 🌍 an environment with a **`baseUrl`** variable for instant host switching;
- 🔎 filtering options moved to **Query Params** (`/api/Players/filter`).

## 🗂 Endpoint coverage

| Resource | Requests |
|----------|:--------:|
| Players (incl. `/api/Players/filter`) | 2 |
| Teams | 3 |
| Tournaments | 2 |
| Advertisements | 2 |
| MatchHistory | 1 |
| Ratings | 1 |
| **Total** | **11** |

## 📁 Structure

```text
CyberSportsPortal API Tests/
├── environments/        # environments (baseUrl variable)
├── Get *.yml            # 11 GET requests with assertions and tests
├── opencollection.yml   # collection manifest (OpenCollection)
└── README.md
```

## 🚀 Quick start

### 1. Install Bruno
Grab the [desktop app or the VS Code extension](https://www.usebruno.com/downloads).

### 2. Clone the repository
```bash
git clone https://github.com/TWOINTHEPINKY/bruno-test-praktika.git
cd bruno-test-praktika
```

### 3. Open the collection
In Bruno: **Open Collection** → select the folder containing `opencollection.yml`.

### 4. Select an environment
Pick one from `environments/` (top-right corner) and adjust `baseUrl` if needed.

### 5. Run the tests
Send a single request (**Send**) or run the whole collection (**Run Collection**).
Assertion and JS-test results are shown in the **Asserts / Tests** tabs.

---

👤 **Danila Fedotov** · SSU · [@TWOINTHEPINKY](https://github.com/TWOINTHEPINKY)
