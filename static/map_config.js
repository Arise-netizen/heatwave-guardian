/**
 * HeatWave Guardian — Centralized Map Configuration
 *
 * Provides a standardized, open-source tile layer configuration using
 * OpenStreetMap standard raster tiles.
 * - 100% free and open-source
 * - Requires ZERO private API keys or access tokens
 * - Fully compatible with Leaflet 1.9.4
 */

(function (window) {
  'use strict';

  const MAP_CONFIG = {
    // Standard OpenStreetMap Tile URL (no API key required)
    tileUrl: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',

    // Standard OpenStreetMap tile options and required attribution
    defaultOptions: {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
      crossOrigin: true
    },

    /**
     * Creates and returns a Leaflet TileLayer instance using standard configuration.
     * @param {Object} [customOptions] - Optional overrides for Leaflet TileLayer options.
     * @returns {L.TileLayer} Leaflet TileLayer instance
     */
    createTileLayer: function (customOptions) {
      if (typeof L === 'undefined') {
        console.error('Leaflet library (L) is not loaded before initializing MAP_CONFIG.createTileLayer()');
        return null;
      }
      const options = Object.assign({}, this.defaultOptions, customOptions || {});
      return L.tileLayer(this.tileUrl, options);
    },

    /**
     * Creates and returns a MapLibre GL JS Style Specification (v8)
     * using OpenStreetMap standard raster tiles.
     * - 100% open-source & keyless
     * - Requires ZERO API keys, tokens, or external billing accounts
     * - Standard OpenStreetMap attribution included in source metadata
     * @param {Object} [customOptions] - Optional style overrides.
     * @returns {Object} MapLibre style JSON specification (v8)
     */
    getMapLibreStyle: function (customOptions) {
      const tileUrl = (customOptions && customOptions.tileUrl) || this.tileUrl;
      const attribution = (customOptions && customOptions.attribution) || this.defaultOptions.attribution;
      const maxZoom = (customOptions && customOptions.maxZoom) || this.defaultOptions.maxZoom;

      return {
        version: 8,
        name: 'OSM-Raster-Keyless',
        sources: {
          'osm-raster-source': {
            type: 'raster',
            tiles: [tileUrl],
            tileSize: 256,
            attribution: attribution,
            maxzoom: maxZoom
          }
        },
        layers: [
          {
            id: 'osm-raster-layer',
            type: 'raster',
            source: 'osm-raster-source',
            minzoom: 0,
            maxzoom: maxZoom
          }
        ]
      };
    }
  };

  // Expose globally
  window.MAP_CONFIG = MAP_CONFIG;
})(typeof window !== 'undefined' ? window : this);
