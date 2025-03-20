from rest_framework import serializers

from .models import Audio, Speaker

class AudioInputSerializer(serializers.ModelSerializer):

    class Meta:
        model = Audio
        fields = ['audio', 'num_speakers', 'use_si']

class SummaryOutputSerializer(serializers.ModelSerializer):

    class Meta:
        model = Audio
        fields = ['audio', 'num_speakers', 'transcript', 'summary', 'created_on']