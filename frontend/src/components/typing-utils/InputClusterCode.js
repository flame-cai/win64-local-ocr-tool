// InputClusterCode.js

export function logCharactersBeforeCursor(input) {
  const cursorPosition = input.selectionStart;
  const currentValue = input.value;
  // Log more characters for debugging multi-char sequences
  console.log({
    '-5': currentValue[cursorPosition - 5],
    '-4': currentValue[cursorPosition - 4],
    '-3': currentValue[cursorPosition - 3],
    '-2': currentValue[cursorPosition - 2],
    '-1': currentValue[cursorPosition - 1]
  });
  return;
}

// --- Character Constants ---
export const HALANT = '\u094D';
export const ZWNJ = '\u200C'; // Zero-Width Non-Joiner
export const ZWJ = '\u200D';  // Zero-Width Joiner
export const NUKTA = '\u093C'; // Combining Dot Below (Nukta)
export const ANUSVARA = '\u0902'; // ं
export const VISARGA = '\u0903'; // ः
export const CANDRABINDU = '\u0901'; // ँ
export const AVAGRAHA = '\u093D'; // ऽ
export const DANDA = '\u0964'; // ।
export const DOUBLE_DANDA = '\u0965'; // ॥
export const OM = '\u0950'; // ॐ
// Add constants for other special characters if keys are assigned
// export const DEVANAGARI_ABBREVIATION_SIGN = '\u0970';
// export const DEVANAGARI_SIGN_HIGH_SPACING_DOT = '\u0971';
// export const DEVANAGARI_SIGN_INVERTED_CANDRABINDU = '\u0900';
// export const DEVANAGARI_STRESS_SIGN_UDATTA = '\u0951';
// export const DEVANAGARI_STRESS_SIGN_ANUDATTA = '\u0952';

// --- Consonant Mappings ---
// Maps single Roman keys directly to Devanagari base consonants
export const singleConsonantMap = {
  'k': 'क', 'g': 'ग', 'c': 'च', 'j': 'ज', 'T': 'ट', 't': 'त', 'D': 'ड',
  'd': 'द', 'N': 'ण', 'n': 'न', 'p': 'प', 'b': 'ब', 'm': 'म', 'y': 'य',
  'r': 'र', 'l': 'ल', 'v': 'व', 'V': 'ङ', 'S': 'ष', 's': 'स', 'h': 'ह',
  'L': 'ळ', 'Y': 'ञ',
  'f': 'फ', // Note: 'f' also used for DANDA in miscMap
  'z': 'ज', // Note: 'z' also used for vowel prefixes
  'q': 'क', // Note: 'q' also used for HALANT in miscMap
};

// Structure: triggerKey: { precedingDevanagariBase: { resultChar: devanagariBase, remove: count } }
export const doubleCharMap = {
  'h': { // Aspirates + sh
    'क': { resultChar: 'ख', remove: 3 }, 'ग': { resultChar: 'घ', remove: 3 },
    'च': { resultChar: 'छ', remove: 3 }, 'ज': { resultChar: 'झ', remove: 3 },
    'ट': { resultChar: 'ठ', remove: 3 }, 'ड': { resultChar: 'ढ', remove: 3 },
    'त': { resultChar: 'थ', remove: 3 }, 'द': { resultChar: 'ध', remove: 3 },
    'प': { resultChar: 'फ', remove: 3 }, 'ब': { resultChar: 'भ', remove: 3 },
    'स': { resultChar: 'श', remove: 3 },
  },
  's': {
    'क': { resultChar: 'क्ष', remove: 3 }, // k + s -> ks (maps to kS = क्ष)
  },
  'S': {
    'क': { resultChar: 'क्ष', remove: 3 }  // k + S -> kS
  },
};

// Structure: triggerKey: { precedingDevSequence: { resultChar: devanagariBase, remove: count } }
export const tripleCharMap = {
  'y': {
    'दन': { resultChar: 'ज्ञ', remove: 5 }, // d + n + y -> dny (ज्ञ)
    'गञ': { resultChar: 'ज्ञ', remove: 5 }, // g + Y + y -> gny (ज्ञ)
    'गन': { resultChar: 'ज्ञ', remove: 5 }, // g + n + y -> gny (ज्ञ)
  },
  'r': {
    'श': { resultChar: 'श्र', remove: 3 }, // sh + r -> shr
  },
};


// --- Vowel Mappings ---
// Dependent Vowels (Matras)
export const dependentVowelMap = {
    'a':'ा', 'e':'े', 'i':'ि', 'o':'ो', 'u':'ु',
    'aa': 'ा', 'ee': 'ी', 'ii': 'ी', 'uu': 'ू', 'oo': 'ू',
    'ai':'ै', 'au':'ौ', 'ou':'ौ',
    'Rri':'ृ', 'RrI':'ॄ', 'Lli':'ॢ', 'LlI':'ॣ',
    'ze':'ॆ', 'zo':'ॊ', 'aE':'ॅ', 'aO':'ॉ',
    'zau':'\u094F', // Kashmiri/Bihari Au Matra
};

// Independent Vowels
export const independentVowelMap = {
    'a':'अ', 'A':'अ', 'i':'इ', 'I':'इ', 'u':'उ', 'U':'उ',
    'e':'ए', 'E':'ए', 'o':'ओ', 'O':'ओ',
    'aa':'आ', 'AA':'आ', 'ii':'ई', 'II':'ई', 'ee':'ई',
    'uu':'ऊ', 'UU':'ऊ', 'oo':'ऊ',
    'ai':'ऐ', 'AI':'ऐ', 'au':'औ', 'AU':'औ', 'ou':'औ',
    'RRi':'ऋ', 'RRI':'ॠ', 'LLi':'ऌ', 'LLI':'ॡ',
    'AE':'ॲ', // Marathi AE
    'AO':'ऑ', // Marathi/Borrowed AO
    // 'aE':'ऍ', // Alternate AE - choose one or handle contextually
    // 'aO':'ऑ', // Alternate AO - choose one or handle contextually
    'zEE':'ऎ', // South Indian Short E
    'zO':'ऒ',  // South Indian Short O
    'zA':'ऄ', // Historic/Regional A
    'zAU':'ॵ', // Historic/Regional Au
};

// Combined lookup for potential vowel starting keys/sequences
export const potentialVowelKeys = new Set([
    'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U',
    'R', 'L', 'z' // Covers Rri, Lli, ze, zo, zau etc.
]);

// Combined map for resolving full vowel sequences
export const combinedVowelMap = { ...dependentVowelMap, ...independentVowelMap };

// --- Vowel Sequence Handling Logic ---
// Map for replacements like i+i -> ii, e+i -> ai, etc.
// Structure: { precedingDevChar: { currentKey: replacementDevChar } }
export const vowelReplacementMap = {
    // Dependent Matra Replacements
    'ि': { 'i': 'ी', 'e': 'ी' }, // short i + i/e -> long ii/ee
    'ु': { 'u': 'ू', 'o': 'ू' }, // short u + u/o -> long uu/oo
    'े': { 'e': 'ी', 'i': 'ै' }, // e + e -> ee, e + i -> ai
    'ो': { 'o': 'ू', 'u': 'ौ', 'i': 'ौ' }, // o + o -> oo, o + u/i -> au
    'ृ': { 'I': 'ॄ', 'i': 'ॄ' }, // Rri + I/i -> RrI
    'ॢ': { 'I': 'ॣ', 'i': 'ॣ' }, // Lli + I/i -> LlI
    'ा': { 'a': 'ा', 'E': 'ॅ', 'O': 'ॉ' }, // aa + a -> aa, aa + E -> aE Candra, aa + O -> aO Candra
    // Independent Vowel Replacements
    'इ': { 'i': 'ई', 'I': 'ई', 'e': 'ई', 'E': 'ई' }, // short I + i/I/e/E -> long II/EE
    'उ': { 'u': 'ऊ', 'U': 'ऊ', 'o': 'ऊ', 'O': 'ऊ' }, // short U + u/U/o/O -> long UU/OO
    'ए': { 'e': 'ई', 'E': 'ई', 'i': 'ऐ', 'I': 'ऐ' }, // E + e/E -> EE, E + i/I -> AI
    'ओ': { 'o': 'ऊ', 'O': 'ऊ', 'u': 'औ', 'U': 'औ' }, // O + o/O -> OO, O + u/U -> AU
    'अ': { 'a': 'आ', 'A': 'आ', 'E': 'ॲ', 'O': 'ऑ'}, // A + a/A -> AA, A + E -> AE(Marathi), A + O -> AO(Marathi)
    'ऋ': { 'I': 'ॠ' }, // RRi + I -> RRI
    'ऌ': { 'I': 'ॡ' }, // LLi + I -> LLI
};


// --- Miscellaneous Mappings ---
export const miscMap = { // VOWEL MODIFIERS(m), HALANT(H), NUKTA(N), NUMBERS, CURRENCY etc.
    // Single Key Modifiers / Symbols
    'M': ANUSVARA,      // 'ं'
    'H': VISARGA,       // 'ः'
    'F': AVAGRAHA,      // 'ऽ'
    'q': HALANT,        // '्' (Explicit Halant ONLY - applies differently than Halant+ZWNJ)
    ' ': ' ',
    '.': '.',           // Period
    'f': DANDA,         // '।', Note: 'f' is also consonant 'फ'
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
    'W': ZWJ,           // '\u200D' (Zero Width Joiner)
    'w': ZWNJ,          // '\u200C' (Zero Width Non-Joiner)

    // Sequences (Handled in handleInput based on last key)
    'MM': CANDRABINDU,  // 'ँ' (Replaces Anusvara)
    '.N': NUKTA,        // '◌़' (Applies to preceding consonant)
    'ff': DOUBLE_DANDA, // '॥' (Replaces Danda)
    'om': OM,           // 'ॐ'

    // --- Keys needing assignment for unmapped chars ---
    // Choose appropriate keys and uncomment/add here if needed
    // Example assignments:
    // '\'': '\u0970', // DEVANAGARI ABBREVIATION SIGN
    // '_': '\u0971',  // DEVANAGARI SIGN HIGH SPACING DOT
    // '^': '\u0900',  // DEVANAGARI SIGN INVERTED CANDRABINDU
    // '+': '\u0951',  // DEVANAGARI STRESS SIGN UDATTA
    // '=': '\u0952',  // DEVANAGARI STRESS SIGN ANUDATTA
};

// Helper Map for simple direct insertions (no context needed beyond the key itself)
// Includes digits, space, period, ZWJ, ZWNJ, Avagraha, and any assigned simple symbols
export const simpleInsertMap = {
    ' ': ' ', '.': '.',
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
    'W': ZWJ, 'w': ZWNJ,
    'F': AVAGRAHA, // Avagraha can usually be inserted directly
    // Add keys for other simple insertions if assigned in miscMap
    // '\'': '\u0970', '_': '\u0971', '^': '\u0900', '+': '\u0951', '=': '\u0952',
};

// --- Sequence Prefix Information ---
// Helps identify potential multi-character sequences
// Structure: { key: potentialNextKey[] }
// ** Define the base object first **
export const sequencePrefixes = {
    // Vowel prefixes
    'R': ['r', 'R', 'i', 'I'], // For Rr, RR, Rri, RRI
    'L': ['l', 'L', 'i', 'I'], // For Ll, LL, Lli, LLI
    'z': ['e', 'o', 'a', 'E', 'A', 'O', 'U'], // For ze, zo, za, zE, zA etc.
    'a': ['a', 'e', 'i', 'u', 'E', 'O'], // For aa, ae, ai, au, aE, aO
    'A': ['A', 'E', 'I', 'O', 'U'], // For AA, AE, AI, AO, AU
    'e': ['e', 'i'], // For ee, ei (ai)
    'E': ['E', 'I'], // For EE, EI (ai)
    'i': ['i', 'e'], // For ii, ie (ee)
    'I': ['I', 'E'], // For II, IE (ee)
    'o': ['o', 'u', 'i'], // For oo, ou (au), oi (au?) - ** Initial definition **
    'O': ['O', 'U', 'I'], // For OO, OU (au), OI (au?)
    'u': ['u', 'o'], // For uu, uo (oo)
    'U': ['U', 'O'], // For UU, UO (oo)

    // Misc prefixes
    '.': ['N'], // For Nukta sequence .N
    'M': ['M'], // For Chandrabindu sequence MM
    'f': ['f'], // For Double Danda sequence ff
    // Add 'A', 'U' prefixes if needed for 'AUM' later
};

// ** Modify the object after definition **
// Add 'm' to the potential keys following 'o' for the 'om' sequence
sequencePrefixes['o'] = [...(sequencePrefixes['o'] || []), 'm'];
// If you were implementing AUM:
// sequencePrefixes['A'] = [...(sequencePrefixes['A'] || []), 'U']; // If A can start AU and AUM
// sequencePrefixes['U'] = [...(sequencePrefixes['U'] || []), 'M']; // If U can start UU and follow A in AUM


// --- Helper Functions ---

// Insert Character Sequence (Generic)
export function insertCharacter(input, devanagariRef, charToInsert, cursorPosition) {
    const currentValue = input.value;
    const newValue =
      currentValue.slice(0, cursorPosition) +
      charToInsert +
      currentValue.slice(cursorPosition);
    const newCursorPosition = cursorPosition + charToInsert.length;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Inserted: ${charToInsert}`);
    logCharactersBeforeCursor(input);
}

// Replace Previous Characters (Generic)
export function replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition) {
    const currentValue = input.value;
    const startReplacePos = cursorPosition - charsToRemove;

    // Ensure we don't go below index 0
    if (startReplacePos < 0) {
        console.error(`replacePreviousChars: Attempting to remove ${charsToRemove} chars from pos ${cursorPosition}.`);
        return; // Or handle differently
    }

    const newValue =
      currentValue.slice(0, startReplacePos) +
      charToInsert +
      currentValue.slice(cursorPosition); // Slice from original cursor pos

    // New cursor position: start of replacement + length of inserted char
    const newCursorPosition = startReplacePos + charToInsert.length;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Replaced ${charsToRemove} chars with ${charToInsert}`);
    logCharactersBeforeCursor(input);
}

// Helper to apply Dependent Vowel (Matra)
export function applyDependentVowel(input, devanagariRef, matra, cursorPosition) {
    const currentValue = input.value;
    // Context assumes: Base (charM3) + Halant (charM2) + ZWNJ (charM1) before cursor
    const baseConsonant = currentValue[cursorPosition - 3];
    const charsToRemove = 3; // Base + Halant + ZWNJ
    const charToInsert = baseConsonant + matra;

    // Use the generic replace function
    replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition);
    console.log(`Applied Matra: ${matra} to ${baseConsonant}`);
}

// Insert Consonant Sequence (Base + Halant + ZWNJ)
export function insertConsonantSequence(input, devanagariRef, baseChar, cursorPosition) {
    const currentValue = input.value;
    const sequence = baseChar + HALANT + ZWNJ;
    const sequenceLength = sequence.length; // Should be 3

    const newValue =
      currentValue.slice(0, cursorPosition) +
      sequence +
      currentValue.slice(cursorPosition);

    const newCursorPosition = cursorPosition + sequenceLength;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Inserted ${baseChar} + Halant + ZWNJ`);
    logCharactersBeforeCursor(input);
}

// Replace previous sequence with new Consonant Sequence (Base + Halant + ZWNJ)
export function replaceConsonantSequence(input, devanagariRef, baseChar, cursorPosition, charsToRemove) {
    const currentValue = input.value;
    const sequence = baseChar + HALANT + ZWNJ;
    const sequenceLength = sequence.length; // Should be 3

    const newValue =
      currentValue.slice(0, cursorPosition - charsToRemove) +
      sequence +
      currentValue.slice(cursorPosition);

    // New cursor position: original position - removed chars + inserted chars
    const newCursorPosition = cursorPosition - charsToRemove + sequenceLength;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Replaced ${charsToRemove} chars with ${baseChar} + Halant + ZWNJ`);
    logCharactersBeforeCursor(input);
}

// Handle insertion of a single consonant character
export function handleSingleConsonant(event, devanagariRef, devanagariChar) {
  const input = event.target;
  const cursorPosition = input.selectionStart;
  const currentValue = input.value;
  const characterRelativeMinus1 = currentValue[cursorPosition - 1];

  // No preventDefault needed here, it's handled in handleInput

  if (characterRelativeMinus1 === ZWNJ) {
    // If ZWNJ is just before cursor (e.g., after explicit H+ZWNJ),
    // replace the ZWNJ with the new consonant sequence.
    replacePreviousChars(input, devanagariRef, 1, devanagariChar + HALANT + ZWNJ, cursorPosition);
    console.log(`Replaced ZWNJ with ${devanagariChar} + Halant + ZWNJ`);

  } else {
    // Standard insertion: Append Consonant + Halant + ZWNJ
    insertConsonantSequence(input, devanagariRef, devanagariChar, cursorPosition);
  }
}

// Apply Nukta to the preceding consonant (C+H+ZWNJ -> C+Nukta+H+ZWNJ)
export function applyNukta(input, devanagariRef, cursorPosition) {
    const currentValue = input.value;
    // Context: Base (charM3) + Halant (charM2) + ZWNJ (charM1)
     // Basic check to prevent errors if context is wrong, though handleInput should verify
    if (cursorPosition < 3 || currentValue[cursorPosition - 1] !== ZWNJ || currentValue[cursorPosition - 2] !== HALANT) {
        console.error("applyNukta called with invalid context.");
        return;
    }
    const baseConsonant = currentValue[cursorPosition - 3];
    const charsToRemove = 3; // Base + Halant + ZWNJ
    // Insert Base + Nukta + Halant + ZWNJ
    const charToInsert = baseConsonant + NUKTA + HALANT + ZWNJ;

    replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition);
    console.log(`Applied Nukta to ${baseConsonant}`);
}