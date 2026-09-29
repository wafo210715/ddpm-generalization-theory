# ISM Report Writing Workflow (v2)

## Section Production Pipeline

For each section (2.2, 2.3, 2.4, 3, 4, 5):

### Step 1: Generate Draft Logic Flow
Output: `output/section-X-logic-flow.md`
- Paragraph-by-paragraph logic flow
- Evidence needed (status: ✅ existing / 🔍 needs verification / ❓ gap)
- Callbacks to other sections
- Core arguments per paragraph
- Visuals needed + producibility

### Step 2: Require Evidence
- Agent identifies all references needed
- Agent asks user for missing citations, PDFs, verification
- User puts PDFs in `relevant/references/` (flat, no subfolders)

### Step 3: Reference List + Updated Logic Flow
Output: `output/reference-list-section-X.md` + updated logic flow
- Agent reads available PDFs, classifies evidence
- Agent updates logic flow based on what evidence actually supports
- Reference list with file names for citation in draft
- No subfolder classification — classification lives in the md files only
- Each PDF read once; reuse across sections

### Step 4: Write First Draft
Output: `output/section-X-text.md`
- Full academic prose
- **Citations use file names** (e.g., `[Doan and Zhang - 2025]`), NOT formal (Author, Year)
- User replaces file names with real Zotero citations later

### Step 5: Proof Read + Logic Flow Validation (AFTER all sections drafted)
- Agent reads all reference PDFs
- Cross-checks claims in draft against actual source content
- Validates logic flow holds up against evidence
- Flags any claims that don't match the source
- User may need to provide notes/assistance for books (too long for OCR)

## PDF Handling Rules
- All reference PDFs in `relevant/references/` (flat)
- Interesting papers in `relevant/interesting-food-for-thoughts/`
- Convert PDF → text using pandoc for journal papers
- For books: if too large or OCR fails, ask user for notes/assistance
- Track which PDFs have been read to avoid re-reading

## File Naming Convention
- `section-X-logic-flow.md` — logic flow + evidence map
- `section-X-text.md` — main body prose (file-name citations)
- `reference-list-section-X.md` — references with file names
