import {
    singleConsonantMap, doubleCharMap, tripleCharMap,
    dependentVowelMap, independentVowelMap, combinedVowelMap,
    potentialVowelKeys, vowelReplacementMap, sequencePrefixes,
    miscMap, simpleInsertMap, // Import new maps
    handleSingleConsonant, insertCharacter, replacePreviousChars,
    applyDependentVowel, insertConsonantSequence, replaceConsonantSequence,
    applyNukta, // Import new helper
    logCharactersBeforeCursor,
    HALANT, ZWNJ, ZWJ, NUKTA, ANUSVARA, VISARGA, CANDRABINDU, DANDA, DOUBLE_DANDA, OM // Import constants
  } from './InputClusterCode'
  
  let lastEffectiveKey = null;
  
  export function handleInput(event, devanagariRef) {
    const key = event.key;
    const input = event.target;
    const cursorPosition = input.selectionStart;
    const currentValue = input.value;
  
    // --- Basic Filtering ---
    if (event.metaKey || event.ctrlKey || event.altKey) {
        console.log("Ignoring Ctrl/Meta/Alt key press");
        return;
    }
  
    let effectiveKey = key;
    if (event.shiftKey && key.length === 1 && !key.match(/[a-zA-Z]/)) {
          // Allow Shift + '.' for Nukta trigger? Or other specific combos?
          // For now, treat Shift + non-letter as potentially ignorable or map explicitly
          if (key === '.') { // Allow Shift + '.' if needed later for something else?
               // effectiveKey = '>'; // Example if Shift+. has meaning
               console.log("Shift + . detected, treating as '.' for now");
               effectiveKey = '.'; // Treat as period for now, can change if needed
          } else {
              console.log("Ignoring Shift + Symbol key press:", key);
              lastEffectiveKey = null;
              return;
          }
     } else if (key.length > 1 && key !== 'Backspace') {
         console.log(`Ignoring functional key: ${key}`);
         lastEffectiveKey = null;
         return;
     } else if (event.shiftKey && key.length === 1 && key.match(/[A-Z]/)) {
         effectiveKey = key; // Uppercase letter
     } else if (key.length === 1 && key.match(/[a-z]/)) {
         effectiveKey = key; // Lowercase letter
     } else if (key === 'Backspace') {
          effectiveKey = 'Backspace';
     } else if (simpleInsertMap[key] !== undefined) { // Check if it's a simple insert key (digit, space, ., etc.)
          effectiveKey = key;
     } else if (key === '`') { // Keep explicit halant trigger
         effectiveKey = '`';
     } else if (key === '.') { // Allow period
          effectiveKey = '.';
     } else {
          console.log(`Key "${key}" might pass to fallback or be ignored`);
          // Decide whether to ignore unmapped symbols or let them pass
          // Let's ignore unmapped symbols for now to avoid unexpected chars
          // You can remove this 'return' to allow them.
          // lastEffectiveKey = null; // Reset if ignoring
          // return;
          effectiveKey = key; // Allow pass-through for now
     }
  
  
    console.log("-------------------------");
    console.log(`Effective Key: ${effectiveKey} (at pos ${cursorPosition}) | Last Key: ${lastEffectiveKey}`);
    console.log("State BEFORE processing:");
    logCharactersBeforeCursor(input);
  
    const charM1 = currentValue[cursorPosition - 1];
    const charM2 = currentValue[cursorPosition - 2];
    const charM3 = currentValue[cursorPosition - 3];
    const charM4 = currentValue[cursorPosition - 4];
    const charM5 = currentValue[cursorPosition - 5];
  
    // --- Explicit Halant + ZWNJ ('`' key) ---
    // Insert HALANT + ZWNJ (useful for controlling conjuncts explicitly)
    if (effectiveKey === '`') {
        event.preventDefault();
        const sequence = HALANT + ZWNJ;
        insertCharacter(input, devanagariRef, sequence, cursorPosition);
        console.log('Inserted explicit halant + ZWNJ');
        lastEffectiveKey = effectiveKey;
        return;
    }
  
    // --- Backspace Handling (Keep existing logic) ---
    if (effectiveKey === 'Backspace') {
        lastEffectiveKey = null; // Reset sequence tracking
        if (charM1 === ZWNJ && charM2 === HALANT && cursorPosition >=3 ) {
            event.preventDefault();
            console.log('Backspace: removing Base/Modifier + Halant + ZWNJ'); // Nukta case C+Nukta+H+ZWNJ needs different handling? No, 3 chars works.
            const newValue = currentValue.slice(0, cursorPosition - 3) + currentValue.slice(cursorPosition);
            devanagariRef.value = newValue; input.value = newValue;
            input.setSelectionRange(cursorPosition - 3, cursorPosition - 3);
            logCharactersBeforeCursor(input); return;
        }
         else if (charM2 === HALANT && cursorPosition >= 2) {
             event.preventDefault();
             const newValue = currentValue.slice(0, cursorPosition - 1) + ZWNJ + currentValue.slice(cursorPosition);
             console.log('Backspace: Removed last char, Inserted ZWNJ after halant (original logic)');
             devanagariRef.value = newValue; input.value = newValue;
             input.setSelectionRange(cursorPosition, cursorPosition);
             logCharactersBeforeCursor(input); return;
         }
        else {
            console.log('Backspace: Default behavior');
            queueMicrotask(() => { devanagariRef.value = input.value; logCharactersBeforeCursor(input); });
            return;
        }
    }
  
    // --- Simple Insertions (Space, Digits, ZWJ, ZWNJ, Period, Avagraha etc.) ---
    if (simpleInsertMap[effectiveKey] !== undefined) {
        event.preventDefault();
        const charToInsert = simpleInsertMap[effectiveKey];
        insertCharacter(input, devanagariRef, charToInsert, cursorPosition);
        // Reset last key for space and punctuation, but maybe not for ZWJ/ZWNJ?
        if (charToInsert === ' ' || charToInsert === '.' || charToInsert === AVAGRAHA ) {
            lastEffectiveKey = null;
        } else {
            lastEffectiveKey = effectiveKey; // Keep sequence potential for digits? or ZWJ/ZWNJ? Let's update.
        }
        return;
    }
  
    // --- Miscellaneous Sequence Handling (MM, ff, .N, om) ---
    let potentialMiscSequence = '';
    let miscSequenceHandled = false;
    if (lastEffectiveKey && sequencePrefixes[lastEffectiveKey]?.includes(effectiveKey)) {
          potentialMiscSequence = lastEffectiveKey + effectiveKey;
          console.log("Potential Misc sequence:", potentialMiscSequence);
  
          // Check for MM (Chandrabindu)
          if (potentialMiscSequence === 'MM' && charM1 === ANUSVARA) {
               event.preventDefault();
               replacePreviousChars(input, devanagariRef, 1, CANDRABINDU, cursorPosition);
               miscSequenceHandled = true;
          }
          // Check for ff (Double Danda) - Note conflict with consonant 'f'
          // Prioritize 'ff' if previous was DANDA.
          else if (potentialMiscSequence === 'ff' && charM1 === DANDA) {
               event.preventDefault();
               replacePreviousChars(input, devanagariRef, 1, DOUBLE_DANDA, cursorPosition);
               miscSequenceHandled = true;
          }
           // Check for .N (Nukta)
           else if (potentialMiscSequence === '.N') {
              // Requires C+H+ZWNJ context
              if (cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT && !potentialVowelKeys.has(charM3) /* Ensure it's a consonant base */ ) {
                   event.preventDefault();
                   applyNukta(input, devanagariRef, cursorPosition); // Use helper
                   miscSequenceHandled = true;
              } else {
                  console.log("Nukta (.N) sequence detected but invalid context.");
                  // Prevent default insertion of 'N'? Or allow 'N'? Let's prevent.
                  event.preventDefault();
                  // Don't set miscSequenceHandled = true, let 'N' be potentially handled later if needed
                  lastEffectiveKey = effectiveKey; // Update last key to N
                  return; // Exit early, nukta cannot be applied here
              }
           }
           // Check for om
           else if (potentialMiscSequence === 'om' && miscMap[potentialMiscSequence]) {
               event.preventDefault();
               insertCharacter(input, devanagariRef, OM, cursorPosition);
               miscSequenceHandled = true;
               lastEffectiveKey = null; // Reset sequence after om
               return; // Handled 'om'
           }
    }
  
    if (miscSequenceHandled) {
          lastEffectiveKey = effectiveKey; // Update last key
          return; // Exit if a misc sequence was handled
    }
  
    // --- Explicit HALANT Insertion ('q' key) ---
    // Only inserts HALANT, potentially removing ZWNJ if present
    if (effectiveKey === 'q') {
        event.preventDefault();
        if (charM1 === ZWNJ && charM2 === HALANT) {
            // We are after C + H + ZWNJ. Replace ZWNJ with H. Net effect: remove ZWNJ.
             replacePreviousChars(input, devanagariRef, 1, '', cursorPosition); // Remove ZWNJ
             console.log("Applied explicit halant (q): Removed ZWNJ after existing Halant.");
        } else if (charM1 === ZWNJ) {
             // After explicit HALANT+ZWNJ (` key). Replace ZWNJ with just HALANT.
             replacePreviousChars(input, devanagariRef, 1, HALANT, cursorPosition);
             console.log("Applied explicit halant (q): Replaced ZWNJ with Halant.");
        }
        else {
            // Insert HALANT after a vowel or a consonant+matra
            insertCharacter(input, devanagariRef, HALANT, cursorPosition);
            console.log("Applied explicit halant (q): Inserted Halant.");
        }
        lastEffectiveKey = effectiveKey;
        return;
    }
  
    // --- Single Anusvara / Visarga ('M', 'H') Application ---
    if (effectiveKey === 'M' || effectiveKey === 'H') {
        event.preventDefault();
        const modifier = (effectiveKey === 'M') ? ANUSVARA : VISARGA;
  
        if (cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT) {
            // Preceded by C + H + ZWNJ. Replace H+ZWNJ with modifier.
            // Base char is charM3
            const baseChar = charM3;
            replacePreviousChars(input, devanagariRef, 2, modifier, cursorPosition); // Remove H+ZWNJ, add modifier
            console.log(`Applied modifier ${modifier} after ${baseChar} (replacing H+ZWNJ)`);
        } else if (cursorPosition > 0 && charM1 !== HALANT) {
             // Preceded by a full character (Vowel or C+Matra). Append modifier.
             insertCharacter(input, devanagariRef, modifier, cursorPosition);
             console.log(`Appended modifier ${modifier} after ${charM1}`);
        } else {
            // Context not suitable (e.g., start of input, after halant without ZWNJ)
            console.log(`Cannot apply modifier ${modifier} in current context.`);
            // Optionally insert with dotted circle: insertCharacter(input, devanagariRef, '\u25CC' + modifier, cursorPosition);
        }
        lastEffectiveKey = effectiveKey;
        return;
    }
  
      // --- Single Danda Insertion ('f' key) ---
      // Needs careful handling due to 'f' also mapping to consonant 'फ'
      // Rule: If 'f' is pressed AND it wasn't part of 'ff', treat as DANDA *unless*
      // the context implies the consonant 'फ'.
      // Let's prioritize DANDA if not after C+H+ZWNJ.
      if (effectiveKey === 'f') {
          const isConsonantContext = cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT;
          const isConsonantReplacementContext = doubleCharMap['h']?.[charM3] !== undefined && charM1 === ZWNJ && charM2 === HALANT; // e.g., p+h -> ph
  
          // If not likely forming 'फ' or 'ph', insert Danda.
          if (!isConsonantContext && !isConsonantReplacementContext) {
              event.preventDefault();
              insertCharacter(input, devanagariRef, DANDA, cursorPosition);
              console.log("Inserted Danda (|)");
              lastEffectiveKey = effectiveKey; // Treat as sequence starter for 'ff'
              return;
          }
          // Otherwise, let it fall through to consonant handling below.
          console.log("'f' key pressed in consonant context, will be handled as 'फ'");
      }
  
  
    // --- Consonant Sequence Completion (Triples, Doubles) ---
    // Keep this logic exactly as it was
    const tripleMappings = tripleCharMap[effectiveKey];
    if (tripleMappings && cursorPosition >= 5) { /* ... triple check logic ... */
        if (charM1 === ZWNJ && charM2 === HALANT && charM4 === HALANT) {
            const precedingSequence = charM5 + charM3;
            if (tripleMappings[precedingSequence]) {
                const mapping = tripleMappings[precedingSequence];
                event.preventDefault();
                replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                lastEffectiveKey = effectiveKey; return;
            }
        }
         if (effectiveKey === 'r' && charM1 === ZWNJ && charM2 === HALANT && charM3 === 'श' && tripleMappings['श']) {
              const mapping = tripleMappings['श'];
              event.preventDefault();
              replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
              lastEffectiveKey = effectiveKey; return;
         }
    }
    const doubleMappings = doubleCharMap[effectiveKey];
    if (doubleMappings && cursorPosition >= 3) { /* ... double check logic ... */
         if (charM1 === ZWNJ && charM2 === HALANT) {
            const precedingBase = charM3;
            if (doubleMappings[precedingBase]) {
                const mapping = doubleMappings[precedingBase];
                event.preventDefault();
                replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                lastEffectiveKey = effectiveKey; return;
            }
        }
    }
  
    // --- Vowel Handling Logic (Keep existing logic) ---
    let potentialVowelSequence = '';
    if (lastEffectiveKey && sequencePrefixes[lastEffectiveKey]?.includes(effectiveKey) && potentialVowelKeys.has(effectiveKey[0])) {
         potentialVowelSequence = lastEffectiveKey + effectiveKey;
          console.log("Potential Vowel sequence:", potentialVowelSequence);
          if (combinedVowelMap[potentialVowelSequence]) {
              const isDependentContext = charM1 === ZWNJ && charM2 === HALANT && cursorPosition >= 3;
              const isVowelReplacementContext = vowelReplacementMap[charM1]?.[effectiveKey];
  
              if (isVowelReplacementContext) {
                  event.preventDefault();
                  const replacementChar = vowelReplacementMap[charM1][effectiveKey];
                  replacePreviousChars(input, devanagariRef, 1, replacementChar, cursorPosition);
                  console.log(`Vowel Replacement: ${charM1} + ${effectiveKey} -> ${replacementChar}`);
                  lastEffectiveKey = effectiveKey; return;
              } else if (isDependentContext && dependentVowelMap[potentialVowelSequence]) {
                  event.preventDefault();
                  applyDependentVowel(input, devanagariRef, dependentVowelMap[potentialVowelSequence], cursorPosition);
                  console.log(`Applied complex matra: ${dependentVowelMap[potentialVowelSequence]}`);
                  lastEffectiveKey = effectiveKey; return;
              } else if (!isDependentContext && independentVowelMap[potentialVowelSequence]) {
                  event.preventDefault();
                  insertCharacter(input, devanagariRef, independentVowelMap[potentialVowelSequence], cursorPosition);
                  console.log(`Inserted complex independent vowel: ${independentVowelMap[potentialVowelSequence]}`);
                  lastEffectiveKey = effectiveKey; return;
              } else {
                   console.log(`Sequence ${potentialVowelSequence} valid but context mismatch?`);
              }
        }
    }
     // Vowel Replacement Check (single key)
     if (potentialVowelKeys.has(effectiveKey) && charM1 && vowelReplacementMap[charM1]?.[effectiveKey]) {
          event.preventDefault();
          const replacementChar = vowelReplacementMap[charM1][effectiveKey];
          replacePreviousChars(input, devanagariRef, 1, replacementChar, cursorPosition);
          console.log(`Vowel Replacement (single key): ${charM1} + ${effectiveKey} -> ${replacementChar}`);
          lastEffectiveKey = effectiveKey; return;
     }
     // Single Vowel / Single Consonant Handling
    const isDepContext = charM1 === ZWNJ && charM2 === HALANT && cursorPosition >= 3;
    const devDep = dependentVowelMap[effectiveKey];
    const devIndep = independentVowelMap[effectiveKey];
    const devCons = singleConsonantMap[effectiveKey];
  
    if (isDepContext) {
        if (devDep) {
            event.preventDefault(); applyDependentVowel(input, devanagariRef, devDep, cursorPosition);
            lastEffectiveKey = effectiveKey; return;
        } else if (devCons) {
            event.preventDefault();
            replacePreviousChars(input, devanagariRef, 1, devCons + HALANT + ZWNJ, cursorPosition);
            console.log(`Forming conjunct: Removed ZWNJ, added ${devCons}+H+ZWNJ`);
            lastEffectiveKey = effectiveKey; return;
        } else if (devIndep) {
             event.preventDefault();
             replacePreviousChars(input, devanagariRef, 2, devIndep, cursorPosition);
             console.log(`WARN: Independent vowel after C+H+ZWNJ. Replaced H+ZWNJ with ${devIndep}`);
             lastEffectiveKey = effectiveKey; return;
        }
    } else {
        if (devIndep) {
            event.preventDefault(); insertCharacter(input, devanagariRef, devIndep, cursorPosition);
            lastEffectiveKey = effectiveKey; return;
        } else if (devCons) {
            // Check if it's 'f' which should have been handled as Danda already if appropriate
            if (effectiveKey === 'f') {
                 // If we reached here, 'f' should be treated as consonant 'फ'
                 event.preventDefault();
                 handleSingleConsonant(event, devanagariRef, devCons);
                 lastEffectiveKey = effectiveKey; return;
            } else {
               // Handle other consonants normally
               event.preventDefault(); handleSingleConsonant(event, devanagariRef, devCons);
               lastEffectiveKey = effectiveKey; return;
            }
        } else if (devDep) {
            event.preventDefault();
            const standaloneMatra = '\u25CC' + devDep;
            insertCharacter(input, devanagariRef, standaloneMatra, cursorPosition);
            console.log(`WARN: Dependent vowel in independent context. Inserted ${standaloneMatra}`);
            lastEffectiveKey = effectiveKey; return;
        }
    }
  
    // --- Handle 'h' as a single consonant if it didn't form a double/triple ---
    if (effectiveKey === 'h' && !doubleMappings?.[charM3] && !tripleMappings?.[charM5+charM3]) {
         event.preventDefault();
         handleSingleConsonant(event, devanagariRef, 'ह');
         lastEffectiveKey = effectiveKey;
         return;
    }
  
  
    // --- Fallback ---
    console.log(`Key "${effectiveKey}" not handled by custom logic. Default behavior might occur.`);
    lastEffectiveKey = effectiveKey; // Update last key even if default occurs
    queueMicrotask(() => {
        devanagariRef.value = input.value;
        logCharactersBeforeCursor(input);
    });
  }