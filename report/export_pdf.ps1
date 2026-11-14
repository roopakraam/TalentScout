# Update the table of contents of a .docx in Microsoft Word and export it as PDF.
#
#   powershell -ExecutionPolicy Bypass -File report\export_pdf.ps1 report\TalentScout_v2_Report.docx
#
# Word opens invisibly, refreshes every field (contents, page numbers), saves the .docx
# and writes a .pdf next to it.
param([Parameter(Mandatory = $true)][string]$DocxPath)

$docx = (Resolve-Path $DocxPath).Path
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $false)
    foreach ($table in $doc.Tables) { $table.AutoFitBehavior(1); $table.AutoFitBehavior(2) }   # fit contents, then page width
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $null = $doc.Fields.Update()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)        # 17 = wdExportFormatPDF
    Write-Output ("pages: " + $doc.ComputeStatistics(2))
    $doc.Close($false)
    Write-Output "saved $pdf"
}
finally {
    try { $word.Quit() } catch { }   # Word sometimes closes itself first
}
