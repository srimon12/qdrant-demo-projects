// GeoSmart Alpine.js Reactive Component
window.geoSmart = function () {
  return {
    // Search State
    query: 'modern stylish apartment with fast wifi',
    mode: 'viewport', // 'viewport' | 'decay' | 'polygon' | 'grouped'
    targetBudget: 95,
    priceScale: 35,
    distScale: 2000,
    showFormulaTuner: false,
    isSearching: false,
    selectedStayId: null,
    error: null,
    _searchTimer: null,

    // Formula Weights
    weights: {
      geo: 1.2,
      budget: 0.8,
      semantic: 0.8,
      rating: 0.5,
      superhost: 0.25,
    },

    // Polygon Presets
    activePolygonKey: 'alexanderplatz',
    polygonPresetName: 'Alexanderplatz',
    polygonVertexCount: 387,
    polygonPresets: [
      { key: 'alexanderplatz', label: 'Alexanderplatz', vertices: 387 },
      { key: 'tiergarten_süd', label: 'Tiergarten', vertices: 277 },
      { key: 'tempelhofer_vorstadt', label: 'Kreuzberg', vertices: 223 },
      { key: 'regierungsviertel', label: 'Regierungsviertel', vertices: 166 },
      { key: 'prenzlauer_berg_süd', label: 'Prenzlauer Berg', vertices: 144 },
      { key: 'reuterstraße', label: 'Neukölln', vertices: 76 },
    ],

    // Telemetry & Results
    hits: [],
    hardHits: [],
    districtFacets: [],
    roomFacets: [],
    boundaryAnalysis: null,
    timing: { qdrant_ms: 0, embed_ms: 0, total_ms: 0 },

    // Leaflet Map & Internal References
    map: null,
    markersLayer: null,
    shapesLayer: null,
    markersMap: {},
    isProgrammaticPan: false,
    targetCentroid: { lat: 52.5219, lon: 13.4132 }, // Alexanderplatz centroid

    // Lifecycle Init
    init() {
      this.initLeaflet();
      this.triggerSearch();
    },

    // Initialize Leaflet Map
    initLeaflet() {
      this.map = L.map('map', {
        center: [52.5219, 13.4132],
        zoom: 13,
        zoomControl: false,
      });

      L.control.zoom({ position: 'bottomright' }).addTo(this.map);

      // Clean OpenStreetMap Tile Layer (Free, Open Data, No API key)
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }).addTo(this.map);

      this.markersLayer = L.layerGroup().addTo(this.map);
      this.shapesLayer = L.layerGroup().addTo(this.map);

      // Map Viewport Drag/Zoom Handler (Mode 1)
      this.map.on('moveend', () => {
        if (this.mode === 'viewport' && !this.isProgrammaticPan) {
          this.triggerSearch();
        }
      });

      // Map Click Handler for Destination Drop-Pin (Mode 2/4)
      this.map.on('click', (e) => {
        if (this.mode === 'decay' || this.mode === 'grouped') {
          this.targetCentroid = { lat: e.latlng.lat, lon: e.latlng.lng };
          this.triggerSearch();
        }
      });
    },

    // Switch Search Mode
    setMode(newMode) {
      this.mode = newMode;
      this.selectedStayId = null;

      if (newMode === 'polygon') {
        this.selectPolygonPreset(this.activePolygonKey);
      } else {
        this.triggerSearch();
      }
    },

    // Preset Search Query
    setPreset(text) {
      this.query = text;
      this.triggerSearch();
    },

    // Input & Slider Events
    onSearchInput() {
      this.triggerSearch();
    },

    onSliderChange() {
      this.triggerSearch();
    },

    // Select Polygon Preset (Mode 3)
    async selectPolygonPreset(key) {
      this.activePolygonKey = key;
      const found = this.polygonPresets.find((p) => p.key === key);
      if (found) {
        this.polygonPresetName = found.label;
        this.polygonVertexCount = found.vertices;
      }
      await this.executePolygonSearch();
    },

    // Master Search Dispatcher (debounced - sliders would otherwise stampede the API)
    triggerSearch() {
      clearTimeout(this._searchTimer);
      this._searchTimer = setTimeout(() => this._dispatchSearch(), 250);
    },

    async _dispatchSearch() {
      this.isSearching = true;
      this.error = null;
      try {
        if (this.mode === 'viewport') {
          await this.executeViewportSearch();
        } else if (this.mode === 'decay') {
          await this.executeDecaySearch();
        } else if (this.mode === 'polygon') {
          await this.executePolygonSearch();
        } else if (this.mode === 'grouped') {
          await this.executeGroupedSearch();
        }
      } catch (err) {
        console.error('Search failure:', err);
        this.error = 'Search failed. Is `uv run python main.py --ui` running, and is Qdrant up?';
        this.hits = [];
      }
      this.isSearching = false;
    },

    // Mode 1: Viewport & In-DB Facets
    async executeViewportSearch() {
      if (!this.map) return;
      const bounds = this.map.getBounds();
      const top_left = [bounds.getNorth(), bounds.getWest()];
      const bottom_right = [bounds.getSouth(), bounds.getEast()];

      const t0 = performance.now();
      try {
        const resp = await fetch('/api/search/viewport', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: this.query,
            top_left,
            bottom_right,
            target_price: parseFloat(this.targetBudget),
            price_scale: parseFloat(this.priceScale),
            limit: 20,
          }),
        });
        const data = await resp.json();
        const networkTotal = (performance.now() - t0).toFixed(1);

        this.timing = {
          qdrant_ms: data.timing.qdrant_ms,
          embed_ms: data.timing.embed_ms,
          total_ms: networkTotal,
        };

        this.hits = data.hits || [];
        this.districtFacets = data.district_facets || [];
        this.roomFacets = data.room_facets || [];
        this.shapesLayer.clearLayers();
        this.renderMapMarkers();
      } catch (err) {
        console.error('Viewport search failure:', err);
        throw err;
      }
    },

    // Mode 2: Multi-Decay vs Boundary Cliff
    async executeDecaySearch() {
      const t0 = performance.now();
      try {
        const resp = await fetch('/api/search/decay', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: this.query,
            center_lat: this.targetCentroid.lat,
            center_lon: this.targetCentroid.lon,
            dist_scale_m: parseFloat(this.distScale),
            target_price: parseFloat(this.targetBudget),
            price_scale: parseFloat(this.priceScale),
            weight_geo: parseFloat(this.weights.geo),
            weight_price: parseFloat(this.weights.budget),
            weight_score: parseFloat(this.weights.semantic),
            weight_rating: parseFloat(this.weights.rating),
            weight_superhost: parseFloat(this.weights.superhost),
            limit: 20,
          }),
        });
        const data = await resp.json();
        const networkTotal = (performance.now() - t0).toFixed(1);

        this.timing = {
          qdrant_ms: data.timing.qdrant_ms,
          embed_ms: data.timing.embed_ms,
          total_ms: networkTotal,
        };

        this.hits = data.hits || [];
        this.hardHits = data.hard_hits || [];
        this.boundaryAnalysis = data.boundary_analysis || null;
        this.error = null;

        // Render Shapes: Hard Circle (Red) vs Smooth Halo (Emerald)
        this.shapesLayer.clearLayers();
        const radiusM = this.distScale * 0.75;

        // Hard circle (Binary cutoff)
        L.circle([this.targetCentroid.lat, this.targetCentroid.lon], {
          radius: radiusM,
          color: '#ef4444',
          weight: 1.5,
          fillColor: '#ef4444',
          fillOpacity: 0.05,
          dashArray: '4, 4',
        }).addTo(this.shapesLayer);

        // Smooth continuous Gaussian halo
        L.circle([this.targetCentroid.lat, this.targetCentroid.lon], {
          radius: this.distScale,
          color: '#10b981',
          weight: 1.5,
          fillColor: '#10b981',
          fillOpacity: 0.08,
        }).addTo(this.shapesLayer);

        this.renderMapMarkers();
      } catch (err) {
        console.error('Decay search failure:', err);
        throw err;
      }
    },

    // Mode 3: Administrative Polygon Pushdown
    async executePolygonSearch() {
      const polyResp = await fetch(`/api/district_polygon/${this.activePolygonKey}`);
      const polyData = await polyResp.json();
      if (!polyData.points) {
        this.error = `Polygon not found for “${this.activePolygonKey}”. Try another district - data snapshots can rename boundaries.`;
        this.hits = [];
        return;
      }
      this.error = null;

      this.polygonVertexCount = polyData.points.length;

      const t0 = performance.now();
      try {
        const resp = await fetch('/api/search/polygon', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: this.query,
            polygon: polyData.points,
            limit: 20,
          }),
        });
        const data = await resp.json();
        const networkTotal = (performance.now() - t0).toFixed(1);

        this.timing = {
          qdrant_ms: data.timing.qdrant_ms,
          embed_ms: data.timing.embed_ms,
          total_ms: networkTotal,
        };

        this.hits = data.hits || [];

        // Render GeoPolygon on Map
        this.shapesLayer.clearLayers();
        const latlngs = polyData.points.map((p) => [p.lat, p.lon]);
        const poly = L.polygon(latlngs, {
          color: '#3b82f6',
          weight: 2,
          fillColor: '#3b82f6',
          fillOpacity: 0.12,
        }).addTo(this.shapesLayer);

        this.isProgrammaticPan = true;
        this.map.fitBounds(poly.getBounds(), { padding: [30, 30] });
        setTimeout(() => {
          this.isProgrammaticPan = false;
        }, 500);

        this.renderMapMarkers();
      } catch (err) {
        console.error('Polygon search failure:', err);
        throw err;
      }
    },

    // Mode 4: Grouped Diversity
    async executeGroupedSearch() {
      const t0 = performance.now();
      try {
        const resp = await fetch('/api/search/grouped', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: this.query,
            center_lat: this.targetCentroid.lat,
            center_lon: this.targetCentroid.lon,
            dist_scale_m: parseFloat(this.distScale),
            target_price: parseFloat(this.targetBudget),
            price_scale: parseFloat(this.priceScale),
            group_size: 1,
            limit: 6,
          }),
        });
        const data = await resp.json();
        const networkTotal = (performance.now() - t0).toFixed(1);
        this.timing = {
          qdrant_ms: data.timing.qdrant_ms,
          embed_ms: data.timing.embed_ms,
          total_ms: networkTotal,
        };
        // Flatten grouped hits for map/cards while preserving district label
        const flat = [];
        (data.groups || []).forEach((g) => {
          (g.hits || []).forEach((h) => {
            h._group = g.district;
            flat.push(h);
          });
        });
        this.hits = flat;
        this.shapesLayer.clearLayers();
        L.circle([this.targetCentroid.lat, this.targetCentroid.lon], {
          radius: this.distScale,
          color: '#10b981',
          weight: 1.5,
          fillColor: '#10b981',
          fillOpacity: 0.06,
        }).addTo(this.shapesLayer);
        this.renderMapMarkers();
      } catch (err) {
        console.error('Grouped search failure:', err);
        throw err;
      }
    },

    // Calculate Normalized Match Percentage (0-100%)
    getMatchPct(rawScore) {
      if (this.mode === 'polygon') {
        // Pure cosine similarity (0.0 to 1.0)
        return Math.min(99, Math.max(10, Math.round(rawScore * 100)));
      }
      // Composite multi-decay formula (theoretical max ~3.80)
      const maxTheoretical = 3.8;
      return Math.min(99, Math.max(10, Math.round((rawScore / maxTheoretical) * 100)));
    },

    // Select / Highlight Stay
    selectStay(hit) {
      this.selectedStayId = hit.id;

      // Smoothly fly map to location and pop open marker
      this.isProgrammaticPan = true;
      this.map.flyTo([hit.lat, hit.lon], 15, { duration: 0.5 });

      setTimeout(() => {
        this.isProgrammaticPan = false;
        const marker = this.markersMap[hit.id];
        if (marker) {
          marker.openPopup();
        }
      }, 550);

      // Scroll card into view
      const targetCard = document.getElementById(`card-stay-${hit.id}`);
      if (targetCard) {
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    },

    // Render Markers on Map
    renderMapMarkers() {
      this.markersLayer.clearLayers();
      this.markersMap = {};
      const hardIds = new Set((this.hardHits || []).map((h) => h.id));

      this.hits.forEach((h, idx) => {
        // Decay mode: amber = recovered from beyond the hard circle, green = also inside it
        const recovered = this.mode === 'decay' && hardIds.size > 0 && !hardIds.has(h.id);
        const marker = L.circleMarker([h.lat, h.lon], {
          radius: idx === 0 ? 8 : 6,
          fillColor: recovered ? '#f59e0b' : '#10b981',
          color: '#ffffff',
          weight: 1.5,
          opacity: 1,
          fillOpacity: 0.9,
        });

        const sh = h.is_superhost ? '<span style="color: #f59e0b; font-weight:600;">★ Superhost</span> · ' : '';
        marker.bindPopup(`
          <div style="font-family: 'Geist', sans-serif; font-size: 12px; line-height: 1.35; padding: 2px;">
            <b style="color: #10b981;">#${idx + 1} ${h.name}</b><br>
            <b>€${Math.round(h.price)}/nt</b> · ★ ${h.rating.toFixed(2)} (${h.number_of_reviews})<br>
            ${sh}<span style="color: #888;">${h.neighbourhood}</span>
          </div>
        `);

        marker.on('click', () => {
          this.selectedStayId = h.id;
          const targetCard = document.getElementById(`card-stay-${h.id}`);
          if (targetCard) {
            targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        });

        this.markersMap[h.id] = marker;
        this.markersLayer.addLayer(marker);
      });
    },
  };
};
