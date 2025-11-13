import { describe, expect, it } from "vitest";

import { kmlDocumentToGeometry, parseKmlOrKmz } from "../kml";

describe("kmlDocumentToGeometry", () => {
  it("converts a simple polygon placemark", () => {
    const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            -1,0,0 -1,1,0 0,1,0 0,0,0 -1,0,0
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
</kml>`;
    const doc = new DOMParser().parseFromString(kml, "application/vnd.google-earth.kml+xml");
    const geometry = kmlDocumentToGeometry(doc);
    expect(geometry).not.toBeNull();
    expect(geometry?.type).toBe("Polygon");
  });
});

describe("parseKmlOrKmz", () => {
  it("reads polygon geometry from a KMZ archive", async () => {
    const base64 =
      "UEsDBBQAAAAIACq7bVsckLHO1wAAAJoBAAAHAAAAZG9jLmttbHVQzWrCQBC+5ymWPWsmeiplEqEUodCD1PYBljjEYDKjuxujb++kVTRQLzv7/TA/Hy5ObWOO5EMtnNtmllDXMqm5iq3P9/L6YtdFAnu1KVODrndxrh/Bej7PpU9cVWHlCmCOmCezm2RGIPvUnYtcRyAwlXjSmqd3/1hZdi1VKwPnfOE8AtuykqacyV8w8pIF8m/Sccb588f4a6o9lkzOf+l2z7SKpQiXo9wkcJYMWY6m2STbCj6Mdn1vVJaxo3gSSeE/2YjPNkWYXSYwsdMEO6J4RBlkVwAUEsBAhQDFAAAAAgAKrttWxyQsc7XAAAAmgEAAAcAAAAAAAAAAAAAAIABAAAAAGRvYy5rbWxQSwUGAAAAAAEAAQA1AAAA/AAAAAAA";
    const buffer = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
    const file = new File([buffer], "square.kmz", { type: "application/vnd.google-earth.kmz" });
    const geometry = await parseKmlOrKmz(file);
    expect(geometry).not.toBeNull();
    expect(geometry?.type).toBe("Polygon");
  });

  it("returns null when no polygon geometry exists", async () => {
    const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <Point>
      <coordinates>0,0,0</coordinates>
    </Point>
  </Placemark>
</kml>`;
    const file = new File([kml], "point.kml", { type: "application/vnd.google-earth.kml+xml" });
    const geometry = await parseKmlOrKmz(file);
    expect(geometry).toBeNull();
  });
});
