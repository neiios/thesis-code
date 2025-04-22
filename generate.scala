//> using scala "3.6.4"
//> using dep com.softwaremill.sttp.client4::core:4.0.3
//> using dep com.softwaremill.sttp.client4::upickle:4.0.3
//> using dep "com.lihaoyi::os-lib:0.11.4"

import sttp.client4.quick.*
import sttp.client4.upicklejson.default.*
import java.time.Instant
import upickle.default.*
import os.{Path, RelPath, pwd}

case class Config(
  baseUrl: String,
  model: String,
  samplesPerRequest: Int,
  snippetsPerTopic: Int,
  topicsFile: String,
  schemaFile: String,
  promptFilesConfig: Map[String, String],
  outputDir: String,
  outputFileTemplate: String,
  requestDelaySeconds: Int,
  apiKey: String = "" // will be set from an env var
) derives ReadWriter

case class SnippetsResponse(snippets: Seq[Snippet]) derives ReadWriter
case class Snippet(code: String) derives ReadWriter

case class OutputLine(
  id: String,
  timestamp: Long,
  code: String,
  category: String,
  topicUsed: String
) derives ReadWriter

case class ApiResponse(choices: Seq[Choice]) derives ReadWriter
case class Choice(message: Message) derives ReadWriter
case class Message(content: String) derives ReadWriter

def loadConfig(configPath: Path): Config =
  if !os.exists(configPath) then throw Exception(s"Config file not found at $configPath")
  val configJson = os.read(configPath)
  val config = read[Config](configJson)
  val apiKey = sys.env.getOrElse("MODEL_API_KEY", throw Exception("MODEL_API_KEY environment variable is not set"))
  config.copy(apiKey = apiKey)

def loadSchema(schemaPath: Path): ujson.Value =
  if !os.exists(schemaPath) then throw Exception(s"Schema file not found at $schemaPath")

  val schemaContent = os.read(schemaPath)
  val schema = read[ujson.Value](schemaContent)
  println(s"Loaded schema from $schemaPath")
  schema

def loadTopics(topicsPath: Path): List[String] =
  if !os.exists(topicsPath) then throw Exception(s"Topics file not found at $topicsPath")

  val topicsContent = os.read(topicsPath)
  val topics = read[List[String]](topicsContent)

  if topics.isEmpty then throw Exception(s"Topics file $topicsPath must contain a non-empty list")

  println(s"Loaded ${topics.size} topics from $topicsPath")
  topics

def countExistingSnippets(filePath: Path, topic: String): Int =
  if !os.exists(filePath) then 0
  else
    val lines = os.read.lines(filePath)
    lines.count { line =>
      try
        val record = read[OutputLine](line)
        record.topicUsed == topic
      catch case _: Exception => false
    }

def generateCodeSnippets(config: Config, prompt: String, schema: ujson.Value, model: String): Seq[Snippet] =
  val requestBody = ujson.Obj(
    "model" -> model,
    "messages" -> ujson.Arr(
      ujson.Obj(
        "role" -> "user",
        "content" -> prompt
      )
    ),
    "response_format" -> ujson.Obj(
      "type" -> "json_schema",
      "json_schema" -> ujson.Obj(
        "name" -> "scala_code_snippets",
        "strict" -> true,
        "schema" -> schema
      )
    )
  )

  val response = quickRequest
    .post(uri"${config.baseUrl}/chat/completions")
    .header("Authorization", s"Bearer ${config.apiKey}")
    .header("Content-Type", "application/json")
    .body(write(requestBody))
    .send()

  try
    val apiResponse = read[ApiResponse](response.body)
    val content = apiResponse.choices.headOption
      .map(_.message.content)
      .getOrElse(throw Exception("No content in API response"))
    val snippetsResponse = read[SnippetsResponse](content)
    snippetsResponse.snippets
  catch
    case e: Exception =>
      throw Exception(s"Failed to parse API response: ${e.getMessage}. Response: ${response.body}")

def processCategory(
  config: Config,
  category: String,
  promptPath: Path,
  topics: List[String],
  schema: ujson.Value,
  outputDir: Path
): Int =
  println(s"\n--- Generating snippets for category: $category ---")

  if !os.exists(promptPath) then
    println(s"Error: Prompt file $promptPath not found. Skipping category $category.")
    return 0

  val templateContent = os.read(promptPath)
  val outputFile = outputDir / config.outputFileTemplate.format(category)

  os.makeDir.all(outputFile / os.up)

  var totalSnippetsGenerated = 0

  for (topic, topicIndex) <- topics.zipWithIndex do
    val existingCount = countExistingSnippets(outputFile, topic)
    if existingCount >= config.snippetsPerTopic then
      println(s"Topic '$topic' already has $existingCount snippets for category $category. Skipping.")
    else
      val snippetsNeeded = config.snippetsPerTopic - existingCount
      var snippetsGenerated = 0

      println(s"Generating snippets for topic '$topic' (${topicIndex + 1}/${topics.size}), need $snippetsNeeded more")

      val outStream = os.write.append.outputStream(outputFile)
      try
        while snippetsGenerated < snippetsNeeded do
          val batchSize = math.min(config.samplesPerRequest, snippetsNeeded - snippetsGenerated)

          println(s"Requesting $batchSize snippets for topic '$topic'...")

          val prompt = templateContent
            .replace("$N", batchSize.toString)
            .replace("$TOPIC", topic)

          val snippets = generateCodeSnippets(config, prompt, schema, config.model)
          println(s"Received ${snippets.size} snippets")

          snippets.zipWithIndex.foreach { case (snippet, index) =>
            val totalSnippets = os.read.lines(outputFile).size
            val record = OutputLine(
              id = s"${category}_${totalSnippets + index + 1}",
              timestamp = Instant.now().toEpochMilli,
              code = snippet.code,
              category = category,
              topicUsed = topic
            )
            val line = write(record) + "\n"
            outStream.write(line.getBytes("UTF-8"))
            snippetsGenerated += 1
          }

          println(s"Progress for topic '$topic': ${existingCount + snippetsGenerated}/${config.snippetsPerTopic}")

          if snippetsGenerated < snippetsNeeded then Thread.sleep(config.requestDelaySeconds * 1000)
      finally
        outStream.close()

      totalSnippetsGenerated += snippetsGenerated
      println(s"Completed topic '$topic' for category $category: Generated $snippetsGenerated new snippets")

  println(s"Completed category $category: Generated $totalSnippetsGenerated total new snippets across all topics")
  totalSnippetsGenerated

def processCategories(
  config: Config,
  schema: ujson.Value,
  topics: List[String]
): Int =
  val outputDir = pwd / RelPath(config.outputDir)
  os.makeDir.all(outputDir)

  config.promptFilesConfig.foldLeft(0) { case (total, (category, promptFilePath)) =>
    try
      val promptPath = pwd / RelPath(promptFilePath)
      val count = processCategory(
        config = config,
        category = category,
        promptPath = promptPath,
        topics = topics,
        schema = schema,
        outputDir = outputDir
      )
      total + count
    catch
      case ex: Exception =>
        println(s"Error processing category $category: ${ex.getMessage}")
        total
  }

@main
def run: Unit =
  val config = loadConfig(pwd / "config.json")
  val schemaPath = pwd / RelPath(config.schemaFile)
  val topicsPath = pwd / RelPath(config.topicsFile)

  os.makeDir.all(schemaPath / os.up)

  val schema = loadSchema(schemaPath)
  val topics = loadTopics(topicsPath)

  val totalExpectedSnippets = config.promptFilesConfig.size * topics.size * config.snippetsPerTopic
  println(
    s"Planning to generate up to $totalExpectedSnippets snippets (${config.promptFilesConfig.size} categories x ${topics.size} topics x ${config.snippetsPerTopic} snippets per topic)"
  )

  val totalGenerated = processCategories(config, schema, topics)
  println(s"\n=== Dataset generation complete. Total new snippets: $totalGenerated ===")
