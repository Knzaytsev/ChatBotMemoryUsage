#!/usr/bin/env python
# pylint: disable=C0116
# This program is dedicated to the public domain under the CC0 license.
# author: M. yusuf Sarıgöz - github.com/monatis

"""
This simple Telegram bot is intended to varify ASR dataset annotations on Telegram.
You need to obtain your own API token from Bot Father on Telegram and make a few adjustments in the capitalized variables below.
"""

import logging
import os
from typing import Any, Dict, List
import json

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

TOKEN = ""
BASE_DIR = 'vicuna_experiments' # change if necessary
ANNOTATION_FILE = os.path.join(BASE_DIR, 'annotation_dataset.json') # file that contains annotations in ljspeech 1.1 format.
ANNOTATED_DATA_FILE = os.path.join(BASE_DIR, 'annotated_data.json') # file to be created to write varified annotations.
START_BTN_TEXT = "Let's get started! 🚀" # change if necessary
ANSWER_A = 'Answer A'
ANSWER_B = 'Answer B'
HELP_TEXT = """Hello!"""

if not os.path.exists(ANNOTATED_DATA_FILE):
    with open(ANNOTATED_DATA_FILE, 'a+') as out_file:
        json.loads([], out_file)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)


logger = logging.getLogger(__name__)

SHOW_HELP, ASK_ANNOTATION = range(2)

def start(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [[START_BTN_TEXT]]

    update.message.reply_text(
        HELP_TEXT,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return SHOW_HELP


def ask_annotation(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    msg = update.message.text
    logger.info("%s: '%s'", user.first_name, msg)
    if msg != START_BTN_TEXT:
        # parse user message and correct annotation accordingly
        id = context.bot_data['cur_id']
        annotation = context.bot_data['annotations'][id]
        
        with open(ANNOTATED_DATA_FILE, 'a+') as out_file:
            out_data = json.loads(out_file.read())
        
            if msg == ANSWER_A:
                out_data.append({'id': id, 'dialog': annotation['dialog'], 'Answer': ANSWER_A})
            elif msg == ANSWER_B:
                out_data.append({'id': id, 'dialog': annotation['dialog'], 'Answer': ANSWER_B})
                
            json.dump(out_data, out_file)

        id += 1
        context.bot_data['cur_id'] = id

    send_annotation(update, context.bot_data['annotations'][context.bot_data['cur_id']])
    
    return ASK_ANNOTATION

def send_annotation(update: Update, annotation: Dict[str, Any]) -> None:
    reply_keyboard = [[ANSWER_A, ANSWER_B]]
    
    update.message.reply_text(
        '\n\n'.join([annotation['dialog'], 'Answer A:\n' + annotation[ANSWER_A], 'Answer B:\n' + annotation[ANSWER_B]]),
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    

def cancel(update: Update, _: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main(annotations: List[Dict[str, Any]]) -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            SHOW_HELP: [MessageHandler(Filters.regex('^Hadi başlayalım!$'), ask_annotation)],
            ASK_ANNOTATION: [MessageHandler(Filters.regex('.*'), ask_annotation)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)
    dispatcher.bot_data['annotations'] = annotations
    dispatcher.bot_data['cur_id'] = 0

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':


    try:
        with open(ANNOTATION_FILE, 'r') as annotation_file:
            annotations = json.loads(annotation_file.read())
            main(annotations)

    except OSError as err:
        logger.error(f"Unable to open metadata file. Searched in {ANNOTATION_FILE}.\n\n" + str(err))