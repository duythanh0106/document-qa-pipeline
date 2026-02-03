import { test, Page } from "@playwright/test";
import fs from "fs";

/* ============================
   CONFIG
============================ */

const URL = "https://chat.urbox.dev/";

// Login selectors
const EMAIL_INPUT = "#email";
const PASSWORD_INPUT = "#password";
const LOGIN_BUTTON = 'button:has-text("Sign in")';

// Chat input selector
const EDITOR_BOX = "div.ProseMirror";

// Account test
const EMAIL = "2khoa@yopmail.com";
const PASSWORD = "khoa2";

// Input + Output JSON
const INPUT_FILE = "rag_eval.json";
const OUTPUT_FILE = "KB_output.json";

// Restart browser after N questions
const MAX_PER_SESSION = 10;

/* ============================
   HELPER FUNCTIONS
============================ */

/**
 * Login vào chatbot
 */
async function login(page: Page) {
  console.log("🔑 Opening login page...");

  await page.goto(URL);

  await page.waitForSelector(EMAIL_INPUT, { timeout: 60000 });

  await page.fill(EMAIL_INPUT, EMAIL);
  await page.fill(PASSWORD_INPUT, PASSWORD);

  await page.click(LOGIN_BUTTON);

  console.log("✅ Logged in, waiting for chat editor...");

  await page.waitForSelector(EDITOR_BOX, { timeout: 60000 });

  console.log("✅ Chat editor ready!");
}

/**
 * Scroll xuống cuối trang chat
 */
async function scrollToBottom(page: Page) {
  await page.evaluate(() => {
    window.scrollTo(0, document.body.scrollHeight);
  });
}

/**
 * Wait bot trả lời xong
 */
async function waitBotDone(page: Page) {
  console.log("⏳ Waiting bot response...");

  const stopBtn = page.locator('button:has-text("Stop")');

  if (await stopBtn.isVisible().catch(() => false)) {
    await stopBtn.waitFor({ state: "hidden", timeout: 180000 });
  }

  await page.waitForTimeout(2000);

  console.log("✅ Bot finished!");
}

/**
 * Send câu hỏi bằng Enter
 */
async function sendQuestion(page: Page, question: string) {
  console.log("✍️ Sending:", question);

  await page.waitForSelector(EDITOR_BOX, { timeout: 60000 });

  await page.click(EDITOR_BOX);

  // Clear input
  await page.keyboard.press("Control+A");
  await page.keyboard.press("Backspace");

  // Type chậm
  await page.keyboard.type(question, { delay: 30 });

  // Enter gửi
  await page.keyboard.press("Enter");

  console.log("📨 Sent!");
}

/* ============================
   EXTRACT ANSWER
============================ */

async function getLatestBotAnswer(page: Page): Promise<string> {
    console.log("📝 Extracting answer...");
  
    const botBlocks = page.locator("div.chat-assistant");
    const count = await botBlocks.count();
  
    if (count === 0) return "❌ No bot messages found";
  
    const lastBot = botBlocks.nth(count - 1);
  
    // wait answer render
    const pTags = lastBot.locator("p[dir='auto']");
    try {
      await pTags.first().waitFor({ timeout: 10000 });
    } catch {
      return "❌ No <p dir='auto'> found";
    }
  
    // ✅ lấy full text như code cũ (không bao giờ Empty)
    let answerText = (await lastBot.innerText()).trim();
  
    // ============================
    // ✅ CLEAN đúng 3 phần thừa
    // ============================
  
    // 1. Remove Thought line
    answerText = answerText.replace(/Thought for less than a second\s*/g, "");
  
    // 2. Remove nguồn ở cuối
    answerText = answerText.replace(/\(Nguồn:.*?\)\s*/gs, "");
  
    // 3. Remove "2 Sources" / "3 Sources"
    answerText = answerText.replace(/\d+\s*Sources?\s*/g, "");
  
    // ✅ trim lại
    answerText = answerText.trim();
  
    return answerText.length > 0 ? answerText : "❌ Empty answer";
  }
  
/* ============================
   EXTRACT SOURCES
============================ */

async function getLatestBotSources(page: Page): Promise<string[]> {
  console.log("📌 Extracting sources...");

  const botBlocks = page.locator("div.chat-assistant");
  const count = await botBlocks.count();

  if (count === 0) return ["❌ No bot blocks found"];

  const lastBot = botBlocks.nth(count - 1);

  // STEP 1: Click button "1 Source"
  const openSourceBtn = lastBot
    .locator("button")
    .filter({ hasText: "Source" })
    .first();
    if ((await openSourceBtn.count()) === 0) {
    return ["❌ No Source button found"];
  }

  console.log("🖱 Clicking Source expand...");
  await openSourceBtn.click();

  // STEP 2: Wait sources appear
  const sourceButtons = lastBot.locator("button[id^='source-']");

  try {
    await sourceButtons.first().waitFor({ timeout: 8000 });
  } catch {
    return ["❌ Source list did not expand"];
  }

  // STEP 3: Extract filenames FULL
  const total = await sourceButtons.count();

  let sources: string[] = [];

  for (let i = 0; i < total; i++) {
    const rawText = (await sourceButtons.nth(i).innerText()).trim();

    const lines = rawText.split("\n").map((l) => l.trim());

    if (lines.length >= 2) {
      sources.push(lines[1]); // filename full
    }
  }

  sources = [...new Set(sources)];

  return sources.length > 0
    ? sources
    : ["❌ No source filename extracted"];
}

/* ============================
   OUTPUT RESUME SUPPORT
============================ */

/**
 * Load output cũ nếu đã tồn tại → resume tiếp
 */
function loadExistingOutput(): any[] {
  if (!fs.existsSync(OUTPUT_FILE)) return [];

  try {
    const raw = fs.readFileSync(OUTPUT_FILE, "utf-8").trim();
    if (!raw) return [];

    return JSON.parse(raw);
  } catch {
    console.log("⚠️ Output file corrupted, reset...");
    return [];
  }
}

/**
 * Save output cộng dồn
 */
function saveOutput(data: any[]) {
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(data, null, 2), "utf-8");
}

/* ============================
   MAIN TEST
============================ */

test("KB JSON Auto Answer + Source Extract (Restart every 10)", async ({
  browser,
}) => {
  test.setTimeout(1800000);

  /* ============================
     STEP 1: Load KB.json
  ============================ */

  console.log("📂 Loading KB.json...");
  const kbData = JSON.parse(fs.readFileSync(INPUT_FILE, "utf-8"));

  console.log("✅ Total questions:", kbData.length);

  /* ============================
     STEP 2: Load existing output.json
  ============================ */

  let output = loadExistingOutput();
  let startIndex = output.length;

  console.log("📌 Resume from question:", startIndex + 1);

  /* ============================
     STEP 3: Loop until done
  ============================ */

  while (startIndex < kbData.length) {
    console.log("\n===============================");
    console.log("🚀 Starting NEW Chat Session...");
    console.log("===============================\n");

    // Open new browser context
    const context = await browser.newContext();
    const page = await context.newPage();

    // Login again
    await login(page);

    // Ask up to MAX_PER_SESSION questions
    for (
      let i = 0;
      i < MAX_PER_SESSION && startIndex < kbData.length;
      i++
    ) {
      const question = kbData[startIndex].question;

      console.log(`\n==============================`);
      console.log(`❓ Question ${startIndex + 1}: ${question}`);
      console.log(`==============================\n`);

      await scrollToBottom(page);

      // Send question
      await sendQuestion(page, question);

      // Wait bot response done
      await waitBotDone(page);

      // Extract Answer FIRST
      const answer = await getLatestBotAnswer(page);

      // Extract Sources AFTER
      const sources = await getLatestBotSources(page);

      console.log("✅ Answer:", answer);
      console.log("✅ Sources:", sources);

      // ✅ Giữ nguyên structure KB.json nhưng không mutate file gốc
const originalObj = kbData[startIndex];

// clone object và chỉ thêm đúng field answer + source
const newObj = {
  ...originalObj,
  answer: answer,
  source: sources,
};

// push object đầy đủ vào output
output.push(newObj);



      // Save output after each question
      saveOutput(output);

      console.log("💾 Saved:", startIndex + 1);

      startIndex++;

      await page.waitForTimeout(3000);
    }

    console.log("♻️ Closing browser session after 10 questions...");
    await context.close();
  }
    console.log("\n✅ DONE! All questions completed.");
  console.log("📌 Output saved in:", OUTPUT_FILE);
});