import { collectionToGeometry } from "../shp";

describe("collectionToGeometry", () => {
  it("returns polygon geometry when single polygon feature", () => {
    const geometry = collectionToGeometry({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [
              [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 0],
              ],
            ],
          },
          properties: {},
        },
      ],
    });

    expect(geometry.type).toBe("Polygon");
  });

  it("aggregates multiple polygons into a multipolygon", () => {
    const geometry = collectionToGeometry({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
          },
          properties: {},
        },
        {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]],
          },
          properties: {},
        },
      ],
    });

    expect(geometry.type).toBe("MultiPolygon");
  });

  it("throws when no polygon geometry present", () => {
    expect(() =>
      collectionToGeometry({
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: null,
            properties: {},
          },
        ],
      })
    ).toThrow("AOI geometry must contain polygons");
  });
});
