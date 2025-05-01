//> using scala "3.6.4"
//> using dep com.softwaremill.sttp.client4::core:4.0.3
//> using dep com.softwaremill.sttp.client4::upickle:4.0.3
//> using dep "com.lihaoyi::os-lib:0.11.4"
//> using dep "com.github.scopt::scopt:4.1.0"
//> using dep "org.virtuslab::scala-yaml:0.3.0"

import sttp.client4.quick.*
import sttp.client4.upicklejson.default.*
import java.time.Instant
import java.util.UUID
import upickle.default.*
import os.{Path, RelPath, pwd}
import scopt.OParser
import org.virtuslab.yaml.*

object Logger:
  val runTimestamp = Instant.now().toEpochMilli
  val logFile = pwd / s"log_${runTimestamp}.txt"

  def info(message: String): Unit =
    os.write.append(logFile, s"[INFO] $message\n")
    println(s"[INFO] $message")

  def error(message: String): Unit =
    os.write.append(logFile, s"[ERROR] $message\n")
    println(s"[ERROR] $message")

case class SnippetsResponse(snippets: Seq[Snippet]) derives ReadWriter
case class Snippet(codeSnippet: String, isIdiomatic: Boolean) derives ReadWriter

case class OutputLine(
  id: UUID,
  timestamp: Long,
  category: String,
  topicUsed: String,
  isIdiomatic: Boolean,
  code: String
) derives ReadWriter

case class CliArgs(
  baseUrl: String = "https://openrouter.ai/api/v1",
  model: String = "",
  samplesPerRequest: Int = 5,
  snippetsPerTopic: Int = 5,
  topicsFile: String = "",
  schemaFile: String = "",
  promptsFile: String = "",
  outputDir: String = "output",
  outputFileTemplate: String = "%s_snippets.jsonl",
  requestDelaySeconds: Int = 1
)

case class PromptsYaml(
  generic: String,
  null_checks: Option[String] = None,
  classes_for_data: Option[String] = None,
  throws: Option[String] = None
) derives YamlCodec

def parseCommandLine(args: Array[String]): Option[CliArgs] = {
  val builder = OParser.builder[CliArgs]
  import builder.*

  val parser = OParser.sequence(
    programName("scala-code-generator"),
    head("Scala Code Snippet Generator", "1.0"),
    opt[String]("base-url")
      .action((url, config) => config.copy(baseUrl = url))
      .text("API base URL"),
    opt[String]("model")
      .required()
      .action((model, config) => config.copy(model = model))
      .text("Model name to use"),
    opt[Int]("samples-per-request")
      .required()
      .action((n, config) => config.copy(samplesPerRequest = n))
      .text("Number of samples to generate per API request"),
    opt[Int]("snippets-per-topic")
      .required()
      .action((n, config) => config.copy(snippetsPerTopic = n))
      .text("Number of snippets to generate per topic"),
    opt[String]("topics-file")
      .required()
      .action((path, config) => config.copy(topicsFile = path))
      .text("Path to topics JSON file"),
    opt[String]("schema-file")
      .required()
      .action((path, config) => config.copy(schemaFile = path))
      .text("Path to schema JSON file"),
    opt[String]("prompts-file")
      .required()
      .action((path, config) => config.copy(promptsFile = path))
      .text("Path to prompts file"),
    opt[String]("output-dir")
      .action((dir, config) => config.copy(outputDir = dir))
      .text("Output directory for generated snippets"),
    opt[String]("output-template")
      .action((template, config) => config.copy(outputFileTemplate = template))
      .text("Template for output filenames, must contain %s for category"),
    opt[Int]("delay-seconds")
      .action((seconds, config) => config.copy(requestDelaySeconds = seconds))
      .text("Delay in seconds between API requests")
  )

  OParser.parse(parser, args, CliArgs())
}

def loadPrompts(promptsFilePath: Path): Map[String, String] = {
  if !os.exists(promptsFilePath) then throw Exception(s"Prompts file not found at $promptsFilePath")

  val yamlContent = os.read(promptsFilePath)
  yamlContent.as[PromptsYaml] match {
    case Right(prompts) =>
      val genericPrompt = prompts.generic
      val result = scala.collection.mutable.Map[String, String]()
      if prompts.null_checks.isDefined then result("null_checks") = genericPrompt + "\n\n" + prompts.null_checks.get
      if prompts.classes_for_data.isDefined then
        result("classes_for_data") = genericPrompt + "\n\n" + prompts.classes_for_data.get
      if prompts.throws.isDefined then result("throws") = genericPrompt + "\n\n" + prompts.throws.get
      result.toMap
    case Left(error) =>
      throw Exception(s"Failed to parse prompts file: $error")
  }
}

def loadSchema(schemaPath: Path): ujson.Value =
  if !os.exists(schemaPath) then throw Exception(s"Schema file not found at $schemaPath")

  val schemaContent = os.read(schemaPath)
  val schema = read[ujson.Value](schemaContent)
  Logger.info(s"Loaded schema from $schemaPath")
  schema

def loadTopics(topicsPath: Path): List[String] =
  if !os.exists(topicsPath) then throw Exception(s"Topics file not found at $topicsPath")

  val topicsContent = os.read(topicsPath)
  val topics = read[List[String]](topicsContent)

  if topics.isEmpty then throw Exception(s"Topics file $topicsPath must contain a non-empty list")

  Logger.info(s"Loaded ${topics.size} topics from $topicsPath")
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

case class ApiResponse(choices: Seq[Choice]) derives ReadWriter
case class Choice(message: Message) derives ReadWriter
case class Message(content: String) derives ReadWriter

def generateCodeSnippets(
  baseUrl: String,
  apiKey: String,
  prompt: String,
  schema: ujson.Value,
  model: String
): Seq[Snippet] =
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
    .post(uri"$baseUrl/chat/completions")
    .header("Authorization", s"Bearer ${apiKey}")
    .header("Content-Type", "application/json")
    .body(write(requestBody))
    .send()

  try
    val apiResponse = read[ApiResponse](response.body)
    val jsonString = apiResponse.choices.headOption
      .map(_.message.content)
      .getOrElse(throw Exception("No content in API response"))
    val snippetResponse = read[SnippetsResponse](jsonString)
    snippetResponse.snippets
  catch
    case e: Exception =>
      Logger.error(s"Response body: ${response.body}")
      throw Exception(s"Failed to parse API response: ${e.getMessage}")

def processCategory(
  args: CliArgs,
  apiKey: String,
  category: String,
  promptTemplate: String,
  topics: List[String],
  schema: ujson.Value,
  outputDir: Path
): Int =
  Logger.info(s"\n--- Generating snippets for category: $category ---")

  val outputFile = outputDir / args.outputFileTemplate.format(category)
  os.makeDir.all(outputFile / os.up)

  var totalSnippetsGenerated = 0
  for (topic, topicIndex) <- topics.zipWithIndex do
    val existingCount = countExistingSnippets(outputFile, topic)
    if existingCount >= args.snippetsPerTopic then
      Logger.info(s"Topic '$topic' already has $existingCount snippets for category $category. Skipping.")
      totalSnippetsGenerated += existingCount
    else
      val snippetsNeeded = args.snippetsPerTopic - existingCount
      var snippetsGenerated = 0
      Logger.info(
        s"Generating snippets for topic '$topic' (${topicIndex + 1}/${topics.size}), need $snippetsNeeded more"
      )

      val outStream = os.write.append.outputStream(outputFile)
      try
        while snippetsGenerated < snippetsNeeded do
          val batchSize = math.min(args.samplesPerRequest, snippetsNeeded - snippetsGenerated)
          Logger.info(s"Requesting $batchSize snippets for topic '$topic'...")

          val prompt = promptTemplate
            .replace("$N", batchSize.toString)
            .replace("$TOPIC", topic)

          val snippets = generateCodeSnippets(args.baseUrl, apiKey, prompt, schema, args.model)
          Logger.info(s"Received ${snippets.size} snippets")

          snippets.foreach { snippet =>
            val record = OutputLine(
              id = UUID.randomUUID(),
              timestamp = Instant.now().toEpochMilli,
              code = snippet.codeSnippet,
              isIdiomatic = snippet.isIdiomatic,
              category = category,
              topicUsed = topic
            )
            val line = write(record) + "\n"
            outStream.write(line.getBytes("UTF-8"))
            snippetsGenerated += 1
          }

          Logger.info(s"Progress for topic '$topic': ${existingCount + snippetsGenerated}/${args.snippetsPerTopic}")
          if snippetsGenerated < snippetsNeeded then Thread.sleep(args.requestDelaySeconds * 1000)
      finally
        outStream.close()

      totalSnippetsGenerated += snippetsGenerated
      Logger.info(s"Completed topic '$topic' for category $category: Generated $snippetsGenerated new snippets")

  Logger.info(s"Completed category $category: Generated $totalSnippetsGenerated total new snippets across all topics")
  totalSnippetsGenerated

def processCategories(
  args: CliArgs,
  apiKey: String,
  prompts: Map[String, String],
  schema: ujson.Value,
  topics: List[String]
): Int =
  val outputDir = pwd / RelPath(args.outputDir)
  os.makeDir.all(outputDir)

  prompts.foldLeft(0) { case (total, (category, promptTemplate)) =>
    try
      val count = processCategory(
        args = args,
        apiKey = apiKey,
        category = category,
        promptTemplate = promptTemplate,
        topics = topics,
        schema = schema,
        outputDir = outputDir
      )
      total + count
    catch
      case ex: Exception =>
        Logger.error(s"Error processing category $category: ${ex.getMessage}")
        total
  }

parseCommandLine(args) match {
  case Some(cliArgs) =>
    val apiKey = sys.env.getOrElse("MODEL_API_KEY", throw Exception("MODEL_API_KEY environment variable is not set"))

    val schemaPath = pwd / RelPath(cliArgs.schemaFile)
    val topicsPath = pwd / RelPath(cliArgs.topicsFile)
    val promptsPath = pwd / RelPath(cliArgs.promptsFile)

    os.makeDir.all(schemaPath / os.up)

    val schema = loadSchema(schemaPath)
    val topics = loadTopics(topicsPath)
    val prompts = loadPrompts(promptsPath)

    if prompts.isEmpty then
      Logger.error("No prompts found in the prompts file")
      System.exit(1)

    val totalExpectedSnippets = prompts.size * topics.size * cliArgs.snippetsPerTopic
    Logger.info(
      s"Planning to generate up to $totalExpectedSnippets snippets (${prompts.size} categories x ${topics.size} topics x ${cliArgs.snippetsPerTopic} snippets per topic)"
    )

    val totalGenerated = processCategories(cliArgs, apiKey, prompts, schema, topics)
    Logger.info(s"=== Dataset generation complete. Total snippets: $totalGenerated ===")

  case None =>
    System.exit(1)
}
