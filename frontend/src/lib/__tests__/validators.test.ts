import { validateDateRange, validateGeometry, validateSeasons } from "../validators";

describe("validators", () => {
  it("validates date order", () => {
    expect(validateDateRange("2024-01-01", "2024-02-01")).toBeNull();
    expect(validateDateRange("2024-02-01", "2024-01-01")).toBe("End date must be on or after start");
  });

  it("validates geometry presence", () => {
    expect(validateGeometry(null)).toBe("AOI geometry is required");
    expect(validateGeometry({ type: "Polygon", coordinates: [] } as any)).toBeNull();
  });

  it("validates seasons", () => {
    expect(
      validateSeasons([
        {
          sowingDate: "2024-05-01",
          harvestDate: "2024-10-01",
        },
      ] as any)
    ).toBeNull();

    expect(
      validateSeasons([
        {
          sowingDate: "2024-05-01",
          harvestDate: "2024-04-01",
        },
      ] as any)
    ).toBe("harvestDate must be after sowingDate");
  });
});
