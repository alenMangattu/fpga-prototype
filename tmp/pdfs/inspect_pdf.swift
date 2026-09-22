import Foundation
import AppKit
import PDFKit

let root = URL(fileURLWithPath: "/Users/computer/dev/verilog/tmp/pdfs")
let pdf = PDFDocument(url: root.appendingPathComponent("build/inference_chip_product_roadmap.pdf"))!
print("Pages: \(pdf.pageCount)")
var allText = ""
for i in 0..<pdf.pageCount {
    let page = pdf.page(at: i)!
    let text = page.string ?? ""
    allText += "\n--- PDF page \(i+1) ---\n" + text
    let lines = text.split(separator: "\n")
    print("Page \(i+1): \(text.split(whereSeparator: { $0.isWhitespace }).count) words; \(lines.prefix(3).joined(separator: " | "))")
}
try allText.write(to: root.appendingPathComponent("extracted.txt"), atomically: true, encoding: .utf8)

let pagesPerSheet = 6
let cellW = 520, cellH = 770
for group in stride(from: 0, to: pdf.pageCount, by: pagesPerSheet) {
    let size = NSSize(width: cellW * 2, height: cellH * 3)
    let sheet = NSImage(size: size)
    sheet.lockFocus()
    NSColor(calibratedWhite: 0.88, alpha: 1).setFill()
    NSRect(origin: .zero, size: size).fill()
    for j in 0..<min(pagesPerSheet, pdf.pageCount-group) {
        let i = group+j
        let url = root.appendingPathComponent(String(format: "render/page-%03d.png", i+1))
        guard let im = NSImage(contentsOf: url) else { fatalError("Missing render: \(url)") }
        let col = j % 2, row = j / 2
        let x = CGFloat(col * cellW + 15)
        let y = CGFloat((2-row) * cellH + 34)
        let scale = min(CGFloat(cellW-30)/im.size.width, CGFloat(cellH-48)/im.size.height)
        im.draw(in: NSRect(x: x, y: y, width: im.size.width*scale, height: im.size.height*scale))
        let label = "PDF page \(i+1)" as NSString
        label.draw(at: NSPoint(x: x, y: CGFloat((2-row)*cellH + 9)), withAttributes: [
            .font: NSFont.systemFont(ofSize: 15), .foregroundColor: NSColor.black])
    }
    sheet.unlockFocus()
    let bitmap = NSBitmapImageRep(data: sheet.tiffRepresentation!)!
    let data = bitmap.representation(using: .png, properties: [:])!
    try data.write(to: root.appendingPathComponent("render/contact-\(group/pagesPerSheet+1).png"))
}
