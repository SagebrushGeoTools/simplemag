"""Internal helpers for CF-convention CRS encoding in xarray Datasets.

The CF-1.8 / GDAL convention for projected data:
- A scalar variable named ``spatial_ref`` (or ``crs``) carries the CRS
  definition in its ``crs_wkt`` attribute.
- Coordinate variables carry ``standard_name``, ``units``, and ``axis``.
- Every data variable carries ``grid_mapping = "spatial_ref"``.

This encoding is understood by rioxarray, GDAL, QGIS, and the webxtile
library.
"""


def _crs_to_wkt(crs_input):
    """Return (wkt_string, epsg_int_or_None) for any CRS specification.

    Accepts EPSG codes as int or string ("EPSG:32632" or 32632),
    WKT strings, or pyproj-compatible authority strings.
    Returns the raw input string as wkt if pyproj is not available.
    """
    if not crs_input:
        return "", None
    try:
        import pyproj
        crs = pyproj.CRS.from_user_input(crs_input)
        return crs.to_wkt(), crs.to_epsg()
    except Exception:
        return str(crs_input), None


def _crs_variable_attrs(crs_wkt, epsg_code):
    """Return attribute dict for the ``spatial_ref`` scalar CRS variable."""
    attrs = {"grid_mapping_name": "unknown"}
    if crs_wkt:
        attrs["crs_wkt"] = crs_wkt
        # Also add the GDAL-style 'spatial_ref' WKT for maximum compatibility
        attrs["spatial_ref"] = crs_wkt
    if epsg_code:
        attrs["epsg_code"] = f"EPSG:{epsg_code}"
        attrs["grid_mapping_name"] = "transverse_mercator"  # common default; wkt is authoritative
    return attrs
