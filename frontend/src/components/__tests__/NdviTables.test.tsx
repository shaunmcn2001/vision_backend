import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { NdviYearlyTable } from "../NdviYearlyTable";
import { NdviMonthlyTable } from "../NdviMonthlyTable";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  vi.restoreAllMocks();
});

function clickButton(label: string) {
  const buttons = Array.from(container.querySelectorAll("button"));
  const button = buttons.find((element) => element.textContent?.includes(label));
  expect(button).toBeDefined();
  act(() => {
    button!.click();
  });
}

test("NdviYearlyTable renders data and exports downloads", async () => {
  const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

  act(() => {
    root.render(
      <NdviYearlyTable
        averages={[
          { year: 2022, meanNdvi: 0.51 },
          { year: 2023, meanNdvi: null }
        ]}
      />
    );
  });

  const tableText = container.textContent ?? "";
  expect(tableText).toContain("2022");
  expect(tableText).toContain("0.510");

  clickButton("Download CSV");
  expect(createObjectURL).toHaveBeenCalledTimes(1);
  const csvBlob = createObjectURL.mock.calls[0][0] as Blob;
  await expect(csvBlob.text()).resolves.toContain("year,mean_ndvi");

  clickButton("Download JSON");
  expect(createObjectURL).toHaveBeenCalledTimes(2);
  const jsonBlob = createObjectURL.mock.calls[1][0] as Blob;
  await expect(jsonBlob.text()).resolves.toContain("\"meanNdvi\":0.51");

  expect(clickSpy).toHaveBeenCalledTimes(2);
});

test("NdviMonthlyTable outputs last-year rows and downloads", async () => {
  const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

  act(() => {
    root.render(
      <NdviMonthlyTable
        rows={[
          {
            label: "ndvi_2024-01",
            formattedLabel: "Jan 2024",
            year: 2024,
            month: 1,
            meanNdvi: 0.34
          },
          {
            label: "ndvi_2024-02",
            formattedLabel: "Feb 2024",
            year: 2024,
            month: 2,
            meanNdvi: null
          }
        ]}
      />
    );
  });

  const tableText = container.textContent ?? "";
  expect(tableText).toContain("Jan 2024");
  expect(tableText).toContain("0.340");

  clickButton("Download CSV");
  expect(createObjectURL).toHaveBeenCalledTimes(1);
  const csvBlob = createObjectURL.mock.calls[0][0] as Blob;
  await expect(csvBlob.text()).resolves.toContain("label,year,month,mean_ndvi");

  clickButton("Download JSON");
  expect(createObjectURL).toHaveBeenCalledTimes(2);
  const jsonBlob = createObjectURL.mock.calls[1][0] as Blob;
  await expect(jsonBlob.text()).resolves.toContain("ndvi_2024-01");
});
