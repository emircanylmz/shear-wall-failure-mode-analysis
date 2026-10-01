import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const sourcePath = path.join(root, "data", "raw", "Shear_Wall_Database.xlsx");
const outputDir = path.join(root, "data", "processed");
const outputPath = path.join(outputDir, "Shear_Wall_Database_clean.xlsx");
const previewDir = path.join(root, "outputs", "workbook_previews");

await fs.mkdir(previewDir, { recursive: true });

const sourceBlob = await FileBlob.load(sourcePath);
const sourceWorkbook = await SpreadsheetFile.importXlsx(sourceBlob);
const sourcePreview = await sourceWorkbook.render({
  sheetName: "Database",
  range: "A1:M25",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(previewDir, "source_database.png"),
  new Uint8Array(await sourcePreview.arrayBuffer()),
);

const sourceSheet = sourceWorkbook.worksheets.getItem("Database");
const sourceValues = sourceSheet.getRange("A1:M394").values;
if (!Array.isArray(sourceValues) || sourceValues.length !== 394) {
  throw new Error(`Unexpected source shape: ${sourceValues?.length ?? "missing"} rows`);
}

const sourceRows = sourceValues.slice(1);
const sourceIdCounts = new Map();
for (const row of sourceRows) {
  const sourceId = row[0];
  sourceIdCounts.set(sourceId, (sourceIdCounts.get(sourceId) ?? 0) + 1);
}

const featureKey = (row) => [
  row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[12], row[11],
].map((value) => String(value)).join("|");
const featureCounts = new Map();
for (const row of sourceRows) {
  const key = featureKey(row);
  featureCounts.set(key, (featureCounts.get(key) ?? 0) + 1);
}

const failureNames = {
  1: "Flexural",
  2: "Shear",
  3: "Flex-Shear",
  4: "Sliding",
};

const cleanHeaders = [
  "RowID",
  "SourceID",
  "SourceIDRepeated",
  "Author",
  "Specimen",
  "FailureMode",
  "FailureModeName",
  "M/Vlw",
  "lw/tw",
  "ρvwFy,vw/fc",
  "ρhwFy,vw/fc",
  "ρvcFy,vc/fc",
  "ρhcFy,hc/fc",
  "P/fcAg",
  "Section",
  "Ab/Ag",
  "FeatureVectorGroupSize",
];

const cleanRows = sourceRows.map((row, index) => [
  index + 1,
  row[0],
  sourceIdCounts.get(row[0]) > 1 ? 1 : 0,
  row[1],
  row[2],
  row[3],
  failureNames[row[3]] ?? "Unknown",
  row[4],
  row[5],
  row[6],
  row[7],
  row[8],
  row[9],
  row[10],
  row[11],
  row[12],
  featureCounts.get(featureKey(row)),
]);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const database = workbook.worksheets.add("Database");
const dictionary = workbook.worksheets.add("Data Dictionary");

const fontFamily = "Arial";
const navy = "#17365D";
const blue = "#D9EAF7";
const paleBlue = "#EEF5FB";
const amber = "#FFF2CC";
const lightGray = "#E7E6E6";
const green = "#E2F0D9";
const red = "#FCE4D6";

summary.showGridLines = false;
summary.tabColor = navy;
summary.getRange("A2:H2").values = [["Shear Wall Database — Temiz Veri Özeti", null, null, null, null, null, null, null]];
summary.getRange("A2:H2").format.font = { name: fontFamily, size: 16, bold: true, color: "#1F1F1F" };
summary.getRange("A3:H3").format.borders = { bottom: { style: "thin", color: navy } };
summary.getRange("A4:B5").values = [
  ["Kaynak dosya", "Shear_Wall_Database.xlsx"],
  ["Temizleme kapsamı", "Tek veri tablosu, benzersiz RowID, veri sözlüğü ve açık kalite bayrakları"],
];
summary.getRange("A4:A5").format.font = { name: fontFamily, size: 10, bold: true };
summary.getRange("B4:B5").format.font = { name: fontFamily, size: 10, italic: true, color: "#595959" };

summary.getRange("A7:B7").values = [["Veri kontrolü", "Değer"]];
summary.getRange("A8:A12").values = [
  ["Kayıt sayısı"],
  ["Yinelenen SourceID bulunan satırlar"],
  ["Aynı özellik vektörünü paylaşan satırlar"],
  ["Model alanlarında eksik hücre"],
  ["Geçersiz FailureMode"],
];
summary.getRange("B8:B12").formulas = [
  ["=COUNTA(Database!$A$2:$A$394)"],
  ["=COUNTIFS(Database!$C$2:$C$394,1)"],
  ["=COUNTIFS(Database!$Q$2:$Q$394,\">1\")"],
  ["=SUM(COUNTBLANK(Database!$D$2:$D$394),COUNTBLANK(Database!$E$2:$E$394),COUNTBLANK(Database!$F$2:$F$394),COUNTBLANK(Database!$H$2:$H$394),COUNTBLANK(Database!$I$2:$I$394),COUNTBLANK(Database!$J$2:$J$394),COUNTBLANK(Database!$K$2:$K$394),COUNTBLANK(Database!$L$2:$L$394),COUNTBLANK(Database!$M$2:$M$394),COUNTBLANK(Database!$N$2:$N$394),COUNTBLANK(Database!$O$2:$O$394),COUNTBLANK(Database!$P$2:$P$394))"],
  ["=COUNTIFS(Database!$F$2:$F$394,\"<1\")+COUNTIFS(Database!$F$2:$F$394,\">4\")"],
];
summary.getRange("A7:B7").format = {
  fill: navy,
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A8:B12").format.font = { name: fontFamily, size: 10 };
summary.getRange("B8:B12").format.numberFormat = "#,##0";
summary.getRange("A7:B12").format.borders = {
  insideHorizontal: { style: "thin", color: lightGray },
  bottom: { style: "thin", color: navy },
};
summary.getRange("A9:B10").format.fill = amber;
summary.getRange("A11:B12").conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0,
  format: { fill: red, font: { bold: true, color: "#9C0006" } },
});

summary.getRange("D7:F7").values = [["FailureMode", "Ad", "Kayıt sayısı"]];
summary.getRange("D8:E11").values = [
  [1, "Flexural"],
  [2, "Shear"],
  [3, "Flex-Shear"],
  [4, "Sliding"],
];
summary.getRange("F8").formulas = [["=COUNTIFS(Database!$F$2:$F$394,D8)"]];
summary.getRange("F8:F11").fillDown();
summary.getRange("D7:F7").format = {
  fill: navy,
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("D8:F11").format.font = { name: fontFamily, size: 10 };
summary.getRange("D8:F11").format.borders = {
  insideHorizontal: { style: "thin", color: lightGray },
  bottom: { style: "thin", color: navy },
};
summary.getRange("F8:F11").format.numberFormat = "#,##0";

summary.getRange("A14:H16").values = [
  ["Kullanım notları", null, null, null, null, null, null, null],
  ["Model girdileri yalnızca Data Dictionary sayfasında Model girdisi olarak işaretlenen alanlardır. Author, Specimen, RowID ve SourceID tahmin girdisi değildir.", null, null, null, null, null, null, null],
  ["Model performansı Author alanına göre grup ayrımlı doğrulanmalıdır. Aynı özellik vektörünü paylaşan kayıtlar da aynı değerlendirme grubunda tutulmalıdır.", null, null, null, null, null, null, null],
];
summary.getRange("A14:H14").format = {
  fill: blue,
  font: { name: fontFamily, size: 10, bold: true, color: navy },
  borders: { preset: "outside", style: "thin", color: navy },
};
summary.getRange("A15:H16").format = {
  font: { name: fontFamily, size: 10, color: "#404040" },
  fill: paleBlue,
  wrapText: true,
  verticalAlignment: "top",
};
summary.getRange("A1:H18").format.font.name = fontFamily;
summary.getRange("A1:A18").format.columnWidth = 52;
summary.getRange("B1:B18").format.columnWidth = 18;
summary.getRange("C1:C18").format.columnWidth = 3;
summary.getRange("D1:D18").format.columnWidth = 14;
summary.getRange("E1:E18").format.columnWidth = 18;
summary.getRange("F1:F18").format.columnWidth = 14;
summary.getRange("G1:H18").format.columnWidth = 12;
summary.getRange("A15:H16").format.rowHeight = 56;

database.showGridLines = false;
database.tabColor = "#5B9BD5";
database.getRange("A1:Q394").values = [cleanHeaders, ...cleanRows];
database.getRange("A1:Q1").format = {
  fill: navy,
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
};
database.getRange("A2:Q394").format.font = { name: fontFamily, size: 9, color: "#1F1F1F" };
database.getRange("A2:C394").format.horizontalAlignment = "center";
database.getRange("F2:G394").format.horizontalAlignment = "center";
database.getRange("O2:O394").format.horizontalAlignment = "center";
database.getRange("Q2:Q394").format.horizontalAlignment = "center";
database.getRange("A2:C394").format.numberFormat = "0";
database.getRange("F2:F394").format.numberFormat = "0";
database.getRange("H2:P394").format.numberFormat = "0.000000";
database.getRange("O2:O394").format.numberFormat = "@";
database.getRange("Q2:Q394").format.numberFormat = "0";
database.getRange("C2:C394").conditionalFormats.add("cellIs", {
  operator: "equal",
  formula: 1,
  format: { fill: amber, font: { bold: true, color: "#7F6000" } },
});
database.getRange("Q2:Q394").conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 1,
  format: { fill: blue, font: { bold: true, color: navy } },
});
database.getRange("F2:F394").dataValidation = {
  rule: { type: "list", values: [1, 2, 3, 4] },
};
database.getRange("O2:O394").dataValidation = {
  rule: { type: "list", values: ["R", "B", "F"] },
};
database.tables.add("A1:Q394", true, "ShearWallData");
database.freezePanes.freezeRows(1);
database.freezePanes.freezeColumns(2);
database.getRange("A1:A394").format.columnWidth = 9;
database.getRange("B1:B394").format.columnWidth = 10;
database.getRange("C1:C394").format.columnWidth = 14;
database.getRange("D1:D394").format.columnWidth = 30;
database.getRange("E1:E394").format.columnWidth = 19;
database.getRange("F1:F394").format.columnWidth = 12;
database.getRange("G1:G394").format.columnWidth = 17;
database.getRange("H1:P394").format.columnWidth = 15;
database.getRange("Q1:Q394").format.columnWidth = 22;
database.getRange("A1:Q1").format.rowHeight = 34;

const dictionaryRows = [
  ["RowID", "Benzersiz temiz kayıt kimliği", "Tamsayı", "Yok", "Hayır", "1–393; temiz dosyada benzersiz"],
  ["SourceID", "Kaynak çalışma kitabındaki kimlik", "Tamsayı", "Yok", "Hayır", "Kaynakta bazı değerler yinelenir; geriye dönük izleme için korunmuştur"],
  ["SourceIDRepeated", "SourceID başka bir satırda da kullanılmışsa 1", "İkili", "Yok", "Hayır", "Kalite bayrağı"],
  ["Author", "Deneyin raporlandığı çalışma/yazar", "Metin", "Yok", "Grup alanı", "Model girdisi değildir; train/test gruplamasında kullanılır"],
  ["Specimen", "Çalışma içindeki numune adı", "Metin", "Yok", "Hayır", "Author ile birlikte numuneyi tanımlar"],
  ["FailureMode", "Göçme modu kodu", "Kategori", "Yok", "Hedef", "1=Flexural, 2=Shear, 3=Flex-Shear, 4=Sliding"],
  ["FailureModeName", "Göçme modu etiketi", "Metin", "Yok", "Hedef etiketi", "FailureMode kodunun okunabilir karşılığı"],
  ["M/Vlw", "Moment-kesme oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["lw/tw", "Duvar uzunluğu-kalınlığı oranı", "Sayısal", "Boyutsuz", "Evet", "Pozitif olmalı"],
  ["ρvwFy,vw/fc", "Düşey gövde donatısı normalize oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["ρhwFy,vw/fc", "Yatay gövde donatısı normalize oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["ρvcFy,vc/fc", "Düşey sınır/köşe donatısı normalize oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["ρhcFy,hc/fc", "Yatay sınır/köşe donatısı normalize oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["P/fcAg", "Normalize eksenel yük oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["Section", "Kesit tipi", "Kategori", "Yok", "Evet", "R, B veya F; nominal değişken"],
  ["Ab/Ag", "Sınır bölgesi alan oranı", "Sayısal", "Boyutsuz", "Evet", "Negatif olmamalı"],
  ["FeatureVectorGroupSize", "Aynı dokuz model girdisini paylaşan kayıt sayısı", "Tamsayı", "Yok", "Hayır", "1 benzersiz; >1 aynı tasarım girdilerini paylaşır"],
];

dictionary.showGridLines = false;
dictionary.tabColor = "#A5A5A5";
dictionary.getRange("A2:F2").values = [["Veri Sözlüğü", null, null, null, null, null]];
dictionary.getRange("A2:F2").format.font = { name: fontFamily, size: 16, bold: true, color: "#1F1F1F" };
dictionary.getRange("A3:F3").format.borders = { bottom: { style: "thin", color: navy } };
dictionary.getRange("A4:F4").values = [["Kaynak", "Shear_Wall_Database.xlsx dosyasındaki Database sayfası; ölçüm değerleri değiştirilmeden aktarılmıştır.", null, null, null, null]];
dictionary.getRange("A4:A4").format.font = { name: fontFamily, size: 10, bold: true };
dictionary.getRange("B4:F4").format.font = { name: fontFamily, size: 10, italic: true, color: "#595959" };
dictionary.getRange("A6:F6").values = [["Alan", "Tanım", "Tür", "Birim", "Analizdeki rol", "Kural / not"]];
dictionary.getRange("A7:F23").values = dictionaryRows;
dictionary.getRange("A6:F6").format = {
  fill: navy,
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
};
dictionary.getRange("A7:F23").format = {
  font: { name: fontFamily, size: 10, color: "#1F1F1F" },
  verticalAlignment: "top",
  wrapText: true,
};
dictionary.getRange("A7:F23").format.borders = {
  insideHorizontal: { style: "thin", color: lightGray },
  bottom: { style: "thin", color: navy },
};
dictionary.getRange("E7:E23").conditionalFormats.add("containsText", {
  text: "Evet",
  format: { fill: green, font: { bold: true, color: "#375623" } },
});
dictionary.freezePanes.freezeRows(6);
dictionary.getRange("A1:A23").format.columnWidth = 26;
dictionary.getRange("B1:B23").format.columnWidth = 45;
dictionary.getRange("C1:C23").format.columnWidth = 14;
dictionary.getRange("D1:D23").format.columnWidth = 14;
dictionary.getRange("E1:E23").format.columnWidth = 18;
dictionary.getRange("F1:F23").format.columnWidth = 54;
dictionary.getRange("A7:F23").format.rowHeight = 34;

workbook.recalculate();

const summaryCheck = await workbook.inspect({
  kind: "table",
  sheetId: summary.sheetId,
  range: "A2:F16",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
  maxChars: 9000,
});
console.log("SUMMARY_CHECK");
console.log(summaryCheck.ndjson);

const databaseCheck = await workbook.inspect({
  kind: "table",
  sheetId: database.sheetId,
  range: "A1:Q8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 17,
  maxChars: 10000,
});
console.log("DATABASE_CHECK");
console.log(databaseCheck.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log("FORMULA_ERRORS");
console.log(formulaErrors.ndjson);

for (const [sheetName, range, fileName] of [
  ["Summary", "A1:H18", "summary.png"],
  ["Database", "A1:Q25", "database.png"],
  ["Data Dictionary", "A1:F23", "data_dictionary.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, fileName),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const savedBlob = await FileBlob.load(outputPath);
const savedWorkbook = await SpreadsheetFile.importXlsx(savedBlob);
const savedInspection = await savedWorkbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 8000,
  tableMaxRows: 4,
  tableMaxCols: 8,
});
console.log("SAVED_WORKBOOK_CHECK");
console.log(savedInspection.ndjson);
const savedErrors = await savedWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "saved workbook formula error scan",
});
console.log("SAVED_WORKBOOK_ERRORS");
console.log(savedErrors.ndjson);
for (const [sheetName, range, fileName] of [
  ["Summary", "A1:H18", "summary.png"],
  ["Database", "A1:Q25", "database.png"],
  ["Data Dictionary", "A1:F23", "data_dictionary.png"],
]) {
  const preview = await savedWorkbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, fileName),
    new Uint8Array(await preview.arrayBuffer()),
  );
}
console.log(`OUTPUT=${outputPath}`);
