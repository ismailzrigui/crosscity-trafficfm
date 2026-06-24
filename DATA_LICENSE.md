# Data License and Source Notice

This repository contains code, article files, and Markdown documentation/reports. Large raw data files and generated non-Markdown artifacts are excluded from Git and must be obtained or regenerated from the original providers.

## Third-Party Sources

| Source | Provider | License or Terms Note |
|---|---|---|
| NYC TLC Yellow Taxi Trip Records | New York City Taxi and Limousine Commission | Public TLC trip records / NYC open data terms; verify final reuse language before journal submission. |
| Chicago Taxi Trips 2024+ | City of Chicago Data Portal | City of Chicago Data Portal terms through Socrata endpoint. |
| TfL Santander Cycle Hire Journey Data | Transport for London | TfL open data terms / Open Government Licence where applicable. |
| OpenStreetMap Morocco readiness indicators | OpenStreetMap contributors through Overpass API | Open Database License (ODbL). |

## Repository Policy

- Raw downloaded source files are not redistributed in Git.
- Checksums and file-level metadata are recorded in `data_manifest.md`.
- Source provenance and citation notes are recorded in `source_registry.md`.
- Markdown tables and reports are included for auditability; generated CSV, figure, JSON, and extracted-text artifacts are excluded from Git.

Users must comply with each upstream provider's terms when rerunning the pipeline or redistributing derived artifacts.
