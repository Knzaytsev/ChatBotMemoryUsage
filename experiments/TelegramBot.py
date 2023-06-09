#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.

"""
First, a few callback functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Example of a bot-user conversation using ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 5):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import MessageLimit
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import os
import json
from typing import Any, Dict, List

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = ""
BASE_DIR = 'vicuna_experiments' # change if necessary
ANNOTATION_FILE = os.path.join(BASE_DIR, 'annotation_dataset.json') # file that contains annotations in ljspeech 1.1 format.
ANNOTATED_DATA_FILE = os.path.join(BASE_DIR, 'annotated_data.jsonl') # file to be created to write varified annotations.
START_BTN_TEXT = "Let's get started! 🚀" # change if necessary
ANSWER_A = 'Answer A'
ANSWER_B = 'Answer B'
SUITABLE_ANSWERS = 'Both answers are relevant'
NON_SUITABLE_ANSWERS = 'None of the answers are relevant'
SKIP_MSG = 'Skip'
HELP_TEXT = """Hello! In this telegram bot you should annotate dialogs. 
Eeach dialog contains summary, facts about persons and two answers of the Assistant.
You should choose one of the option provided in the keyboard or skip the dialog."""

SHOW_HELP, ASK_ANNOTATION = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [[START_BTN_TEXT]]

    await update.message.reply_text(
        HELP_TEXT,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    
    context.user_data['cur_id'] = 0

    return SHOW_HELP


async def annotate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    msg = update.message.text
    logger.info("%s: '%s'", user.username, msg)
    if msg != START_BTN_TEXT:
        # parse user message and correct annotation accordingly
        id = context.user_data['cur_id']
        annotation = context.bot_data['annotations'][id]
        
        if msg == SKIP_MSG:
            logger.info("%s has skipped dialog %s'", user.username, id)
        elif msg in [ANSWER_A, ANSWER_B, SUITABLE_ANSWERS, NON_SUITABLE_ANSWERS]:             
            with open(ANNOTATED_DATA_FILE, 'a+') as out_file:  
                user_reply = {'id': id, 'dialog': annotation['dialog'], 
                              'Contexted Answer': annotation['Contexted Answer'], 'user': user.username}
                user_reply['Answer'] = msg
                out_file.write(json.dumps(user_reply) + '\n')

        id += 1
        context.user_data['cur_id'] = id

    if context.user_data['cur_id'] == len(context.bot_data['annotations']):
        return await end_annotation(update, context)
        
    reply_keyboard = [[ANSWER_A, ANSWER_B], [SKIP_MSG]]
    
    annotation = context.bot_data['annotations'][context.user_data['cur_id']]
    
    message_text = '\n\n'.join([f"Progress: {context.user_data['cur_id'] + 1}/{len(context.bot_data['annotations'])}", 
                     annotation['dialog'], 
                     'Please choose the most relevant answer.\nThe most relevant answer should follow the next rules:\n1. The answer should not contradict with context, facts and dialog.\n2. The answer should be natural.\n3. The answer should not be absurd or meaningless.\nAnswer A:\n' +
                     annotation[ANSWER_A], 
                     'Answer B:\n' + annotation[ANSWER_B]])
    for x in range(0, len(message_text), MessageLimit.MAX_TEXT_LENGTH):
        await update.message.reply_text(message_text[x:x+MessageLimit.MAX_TEXT_LENGTH],
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
    
    return ASK_ANNOTATION

async def end_annotation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s has annotated all data.", user.username)
    await update.message.reply_text(
        "Congratulations! You have done annotation, thank you!", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.username)
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main(annotations: List[Dict[str, Any]]) -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SHOW_HELP: [MessageHandler(filters.Regex(f'^{START_BTN_TEXT}$'), annotate)],
            ASK_ANNOTATION: [MessageHandler(filters.Regex(f'^({ANSWER_A}|{ANSWER_B}|{SKIP_MSG}|{SUITABLE_ANSWERS}|{NON_SUITABLE_ANSWERS})$'),
                                            annotate)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.bot_data['annotations'] = annotations
    # application.bot_data['cur_id'] = 0

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    try:
        with open(ANNOTATION_FILE, 'r') as annotation_file:
            annotations = json.loads(annotation_file.read())
        main(annotations)

    except OSError as err:
        logger.error(f"Unable to open metadata file. Searched in {ANNOTATION_FILE}.\n\n" + str(err))