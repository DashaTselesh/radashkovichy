# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a historical map visualization project for Radashkovichy from 1888. It displays georeferenced historical land plots, streets, buildings, and property ownership data overlaid on modern OpenStreetMap tiles using Leaflet.js.

The map is a single-page application (no build process) that loads GeoJSON data files and renders them with interactive features.

## Architecture

### Core Technology Stack
- **Leaflet.js 1.9.4**: Interactive map rendering (loaded via CDN)
- **OpenStreetMap**: Base tile layer
- **GeoJSON**: Geographic data format for all map features
- **Vanilla JavaScript**: No frameworks or build tools

### File Structure
```
/
├── index.html           # Single-page application with embedded CSS and JavaScript
└── data/                # GeoJSON data files
    ├── map1-polygons.geojson      # Land plots (участки земли)
    ├── map1-buildings.geojson     # Buildings (дома)
    ├── map1-buildings2.geojson    # Building subdivisions (части участков)
    └── map1-lines.geojson         # Street boundaries (границы улиц)
```

## Data Layer System

The application loads four GeoJSON layers, each styled differently:

### 1. Land Plots (`map1-polygons.geojson`)
- **Color**: Light green (#90EE90)
- **Properties**:
  - `Field_Number`: Plot number
  - `First and last name of the owners`: Property owner names
  - `Length of land plots`: Plot length
  - `Width of land plots`: Plot width
  - `layer_order`: Z-index for rendering order
- **Display**: Shows field numbers as labels at polygon centers

### 2. Buildings (`map1-buildings.geojson`)
- **Color**: Gold (#FFD700)
- **Purpose**: Represents house structures (дома)

### 3. Building Subdivisions (`map1-buildings2.geojson`)
- **Color**: Pink (#FFB6C1)
- **Purpose**: Represents parts of plots (части участков)

### 4. Street Boundaries (`map1-lines.geojson`)
- **Color**: Gray (#808080)
- **Properties**:
  - `street name`: Name of the street
- **Display**: Street names shown as labels with white text shadow for readability

## Map Configuration

**Map Center**: [54.154854, 27.240569] (Radashkovichy, Belarus)
**Initial Zoom**: 18
**Coordinate System**: WGS84 (CRS84)

## Development Workflow

### Running the Application
Simply open [index.html](index.html) in a web browser. No build process or server is required.

For development with live reload, you can use:
```bash
python -m http.server 8000
# or
npx serve
```

### Working with GeoJSON Data

The GeoJSON files contain feature collections with polygon or line geometries. Each feature has:
- **geometry**: Coordinates in [longitude, latitude] format
- **properties**: Metadata like owner names, dimensions, colors, etc.

When modifying data:
1. Ensure coordinates are in WGS84 format (longitude, latitude)
2. Preserve the `layer_order` property to maintain rendering hierarchy
3. Color and styling properties can be overridden per-feature via properties:
   - `fill_color`, `fill_opacity`
   - `stroke_color`, `stroke_width`
   - `dash_array` (for dashed lines)

### Styling System

Three polygon styling functions exist in [index.html](index.html):
- `polygons1Style()` - for land plots (green)
- `polygons2Style()` - for building subdivisions (pink)
- `polygons3Style()` - for buildings (gold)
- `lineStyle()` - for street boundaries (gray)

Each reads style properties from GeoJSON features with fallback defaults.

## Interactive Features

- **Hover effects**: Polygons increase opacity and stroke weight on mouseover
- **Popups**: Click features to see property details
- **Layer control**: Toggle layers on/off via Leaflet layer control (top right)
- **Legend**: Bottom right shows color-coded feature types
- **Scale**: Bottom left displays metric scale

## Language Notes

The codebase contains mixed Russian and English:
- **UI text in HTML/JS**: Russian (comments, popup labels, console logs)
- **Property names in GeoJSON**: English (e.g., "First and last name of the owners")
- **Property values**: Mixed (owner names in Russian/transliterated, measurements in numbers)

When modifying code, maintain this language convention for consistency.
