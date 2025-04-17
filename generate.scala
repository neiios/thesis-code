//> using scala "3.6.4"
//> using dep com.softwaremill.sttp.client4::core:4.0.3
//> using dep com.softwaremill.sttp.client4::upickle:4.0.3
//> using dep "com.lihaoyi::os-lib:0.11.4"

import sttp.client4.quick.*
import sttp.client4.upicklejson.default.*
import java.time.Instant
import java.util.Random
import upickle.default.*
import os.{Path, RelPath, pwd}

case class Config(
  baseUrl: String,
  model: String,
  samplesPerRequest: Int,
  totalSamplesPerCategory: Int,
  randomSeed: Int,
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
  if !os.exists(configPath) then
    throw Exception(s"Config file not found at $configPath")
  val configJson = os.read(configPath)
  val config = read[Config](configJson)
  val apiKey = sys.env.getOrElse("MODEL_API_KEY", 
    throw Exception("MODEL_API_KEY environment variable is not set"))
  config.copy(apiKey = apiKey)

def loadSchema(schemaPath: Path): ujson.Value =
  if !os.exists(schemaPath) then
    throw Exception(s"Schema file not found at $schemaPath")

  val schemaContent = os.read(schemaPath)
  val schema = read[ujson.Value](schemaContent)
  println(s"Loaded schema from $schemaPath")
  schema

def loadTopics(topicsPath: Path): List[String] =
  if !os.exists(topicsPath) then
    throw Exception(s"Topics file not found at $topicsPath")

  val topicsContent = os.read(topicsPath)
  val topics = read[List[String]](topicsContent)
  
  if topics.isEmpty then
    throw Exception(s"Topics file $topicsPath must contain a non-empty list")
  
  println(s"Loaded ${topics.size} topics from $topicsPath")
  topics

def countExistingSnippets(filePath: Path): Int =
  if !os.exists(filePath) then 0
  else os.read.lines(filePath).size
    
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
    val content = apiResponse.choices.headOption.map(_.message.content)
      .getOrElse(throw Exception("No content in API response"))
    val snippetsResponse = read[SnippetsResponse](content)
    snippetsResponse.snippets
  catch
    case e: Exception => 
      throw Exception(s"Failed to parse API response: ${e.getMessage}. Response body: $response.body")

def processCategory(
  config: Config, 
  category: String, 
  promptPath: Path, 
  topics: List[String], 
  schema: ujson.Value, 
  random: Random,
  outputDir: Path
): Int =
  println(s"\n--- Generating snippets for category: $category ---")
  
  if !os.exists(promptPath) then
    println(s"Error: Prompt file $promptPath not found. Skipping category $category.")
    return 0
  
  val templateContent = os.read(promptPath)
  
  val outputFile = outputDir / config.outputFileTemplate.format(category)
  
  val existingCount = countExistingSnippets(outputFile)
  if existingCount >= config.totalSamplesPerCategory then
    println(s"Category $category already has $existingCount snippets. Skipping.")
    return 0
  
  if existingCount > 0 then
    println(s"Resuming category $category: $existingCount snippets already exist")
  
  var snippetsGenerated = 0
  
  os.makeDir.all(outputFile / os.up)
  
  val outStream = os.write.append.outputStream(outputFile)
  try
    while existingCount + snippetsGenerated < config.totalSamplesPerCategory do
      val remaining = config.totalSamplesPerCategory - (existingCount + snippetsGenerated)
      val batchSize = math.min(config.samplesPerRequest, remaining)
      
      val topic = topics(random.nextInt(topics.size))
      println(s"Generating $batchSize snippets for topic '$topic'...")
      
      val prompt = templateContent
        .replace("$N", batchSize.toString)
        .replace("$TOPIC", topic)
      
      val snippets = generateCodeSnippets(config, prompt, schema, config.model)
      println(s"Received ${snippets.size} snippets")
      
      snippets.zipWithIndex.foreach { case (snippet, index) =>
        val record = OutputLine(
          id = s"${category}_${existingCount + snippetsGenerated + index + 1}",
          timestamp = Instant.now().toEpochMilli,
          code = snippet.code,
          category = category,
          topicUsed = topic
        )
        val line = write(record) + "\n"
        outStream.write(line.getBytes("UTF-8"))
        snippetsGenerated += 1
      }
      
      val total = existingCount + snippetsGenerated
      println(s"Progress: $total/${config.totalSamplesPerCategory} for $category")
    
      if existingCount + snippetsGenerated < config.totalSamplesPerCategory then
        Thread.sleep(config.requestDelaySeconds * 1000)
  finally
    outStream.close()
  
  println(s"Completed $category: Generated $snippetsGenerated new snippets")
  snippetsGenerated

def processAllCategories(
  config: Config, 
  schema: ujson.Value, 
  topics: List[String], 
  random: Random,
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
        random = random,
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
  val configPath = pwd / "config.json"
  
  val config = loadConfig(configPath)
  println(s"Using random seed: ${config.randomSeed}")
  val random = new Random(config.randomSeed)
  
  val schemaPath = pwd / RelPath(config.schemaFile)
  val topicsPath = pwd / RelPath(config.topicsFile)
  
  os.makeDir.all(schemaPath / os.up)
  
  val schema = loadSchema(schemaPath)
  val topics = loadTopics(topicsPath)
  val totalGenerated = processAllCategories(config, schema, topics, random)
  println(s"\n=== Dataset generation complete. Total new snippets: $totalGenerated ===")
